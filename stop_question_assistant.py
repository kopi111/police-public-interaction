"""An assistant that answers plain-English questions about Seattle Terry Stops.

The language model only reads the question and writes the reply. Python works out every number from the real data,
and any reply containing a number the data did not produce is replaced by the plain calculated answer.
Everything runs on this computer through Ollama.
"""
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from scipy.special import expit, logit

# Where the project's files are.
PROJECT_FOLDER = Path(__file__).parent
RAW_DATA_PATH = PROJECT_FOLDER / "data/raw/terry_stops.csv"
MODEL_PATH = PROJECT_FOLDER / "models/arrest_outcome_model.joblib"
MODEL_DETAILS_PATH = PROJECT_FOLDER / "models/arrest_outcome_model_details.json"

# The local language model and how to reach it.
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LANGUAGE_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
LANGUAGE_MODEL_TIMEOUT_SECONDS = 300

# Dates the assistant needs in order to understand "this year" and "last year".
FIRST_YEAR = 2015
CURRENT_YEAR = 2026

# Rates from fewer stops than this are too unreliable to rank or to give without a warning.
MINIMUM_STOPS_FOR_A_RATE = 30
MOST_ROWS_TO_SHOW = 10

# Everyday words mapped to the wording used in the data's call types.
CALL_WORD_SYNONYMS = {"ASSAULT": "ASLT", "ASSAULTS": "ASLT", "BURGLARY": "BURG", "BURGLARIES": "BURG",
                      "DOMESTIC": "DV", "DRUG": "NARCOTICS", "DRUGS": "NARCOTICS", "SHOPLIFTING": "SHOPLIFT",
                      "ROBBERIES": "ROBBERY", "WARRANTS": "WARRANT", "WEAPON": "WEAPN", "WEAPONS": "WEAPN", "GUN": "WEAPN",
                      "CRASH": "MVC", "COLLISION": "MVC", "DRUNK": "DUI", "FIGHTS": "FIGHT", "SHOOTING": "SHOT",
                      "SHOOTINGS": "SHOT", "SUSPICIOUS": "SUSPICIOUS", "THEFTS": "THEFT", "STEALING": "THEFT"}

# The outcome that counts as an arrest.
ARREST_OUTCOME = "Arrest"

# Words showing that a question wants groups compared. "By" alone is not enough, as in "called in by 911".
COMPARISON_WORDS = r"\b(which|what \w+ (has|have|had)|most|least|fewest|highest|lowest|compare|comparison|breakdown|trend|over time|top|busiest)\b"

# Words that name each way of grouping, as in "by race", "each year" or "which precinct".
GROUP_WORDS = {"year": "year", "years": "year", "precinct": "precinct", "precincts": "precinct", "race": "race", "gender": "gender",
               "age": "age_group", "outcome": "outcome", "result": "outcome", "call": "call", "hour": "hour", "time": "hour",
               "day": "weekday", "weekday": "weekday", "month": "month"}
GROUP_PATTERN = r"\b(?:by|each|per|every|which|what|across|busiest)\s+(?:perceived\s+|type\s+of\s+)?(\w+)"

# The details below are read by Python, because the language model often drops them.
FILTER_PHRASES = {
    "precinct": [(r"\b(south\s?west)\b", "Southwest"), (r"\bwest\b", "West"), (r"\bnorth\b", "North"), (r"\beast\b", "East"), (r"\bsouth\b", "South")],
    "gender": [(r"\b(women|woman|females?|girls?)\b", "Female"), (r"\b(men|man|males?|boys?)\b", "Male")],
    "race": [(r"\b(black|african american)\b", "Black or African American"), (r"\bwhite\b", "White"), (r"\basians?\b", "Asian"),
             (r"\b(hispanics?|latinos?|latinas?)\b", "Hispanic"), (r"\b(native american|american indian)", "American Indian or Alaska Native"),
             (r"\bpacific islander", "Native Hawaiian or Other Pacific Islander"), (r"\bmulti-?racial\b", "Multi-Racial")],
    "age_group": [(r"\b(under 18|children|child|minors?|juveniles?|teenagers?|teens?)\b", "1 - 17"), (r"\b18\s*(to|-)\s*25\b", "18 - 25"),
                  (r"\b26\s*(to|-)\s*35\b", "26 - 35"), (r"\b36\s*(to|-)\s*45\b", "36 - 45"), (r"\b46\s*(to|-)\s*55\b", "46 - 55"),
                  (r"\b(over 55|56 and (above|over)|seniors?|elderly)\b", "56 and Above")],
    "outcome": [(r"\b(citations?|infractions?)\b", "Citation / Infraction"), (r"\bfield contacts?\b", "Field Contact"),
                (r"\boffen[sc]e reports?\b", "Offense Report"), (r"\b(referred for )?prosecution\b", "Referred for Prosecution"),
                (r"\barrest(s|ed)?\b", ARREST_OUTCOME)],
    "how_received": [(r"\b911\b", "911"), (r"\b(on[- ]?view|officer[- ]initiated)\b", "ONVIEW"), (r"\btelephone\b", "TELEPHONE OTHER, NOT 911"),
                     (r"\balarm calls?\b", "ALARM CALL (NOT POLICE ALARM)")],
}

# Words that describe a stop's details rather than a type of call.
NOT_CALL_WORDS = {"911", "ON", "VIEW", "ARREST", "ARRESTS", "ARRESTED", "FRISK", "FRISKS", "FRISKED", "CITATION", "CITATIONS", "ONVIEW", "PERSON",
                  "PEOPLE", "POLICE", "OFFICER", "OFFICERS", "COPS", "TERRY", "RATE", "CHANCE", "NORTH", "SOUTH", "EAST", "WEST",
                  "SOUTHWEST", "PRECINCT", "BEAT", "WOMEN", "MEN", "FOUND"}

# Words that say nothing about the type of call.
CALL_FILLER_WORDS = {"A", "AN", "THE", "OF", "FOR", "CALL", "CALLS", "STOP", "STOPS", "VIOLENCE", "INCIDENT", "INCIDENTS", "CASE", "CASES"}

# Words in a reply that claim a cause or say the facts were missing, which the facts never support.
BANNED_REPLY_PHRASES = ("because", "due to", "caused by", "as a result", "facts", "does not provide", "do not provide",
                        "not provided", "should")

# The wording of the data's answer columns.
OUTCOMES = ["Arrest", "Field Contact", "Offense Report", "Citation / Infraction", "Referred for Prosecution"]
PRECINCTS = ["West", "North", "East", "South", "Southwest"]
RACES = ["White", "Black or African American", "Asian", "American Indian or Alaska Native", "Hispanic", "Multi-Racial",
         "Native Hawaiian or Other Pacific Islander", "Other", "Unknown"]
GENDERS = ["Male", "Female", "Gender Diverse (gender non-conforming and/or transgender)", "Unable to Determine", "Unknown"]
AGE_GROUPS = ["1 - 17", "18 - 25", "26 - 35", "36 - 45", "46 - 55", "56 and Above"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]

# The name of each rate, as shown in answers.
MEASURE_NAMES = {"arrest_rate": "Arrest rate", "frisk_rate": "Frisk rate", "weapon_rate": "Weapon found rate"}

# Weapon entries that mean no weapon was recorded.
NO_WEAPON_ENTRIES = {"-", "None", "None/Not Applicable"}

# Fixed facts about the project, used for questions about the data and the AI itself.
PROJECT_FACTS = {
    "data": "The data is Seattle Terry Stops from the City of Seattle, public domain, downloaded 2026-11-04. It records "
            "police stops of people under Terry v. Ohio from 2015 to 3 November 2026. Race and gender are as perceived by "
            "the officer. It only covers recorded Terry stops in Seattle, not all police contact and not Jamaica.",
    "accuracy": "The arrest model scored AUC 0.637 on 6,889 stops from 2025-2026, where 0.50 is a coin toss. When it said "
                "likely arrest it was right 28.1% of the time, and it caught 48.9% of real arrests. It is weak.",
    "fairness": "The arrest model never saw race. Among stops with no arrest, it wrongly flagged 32.4% of people perceived "
                "as Black and 25.0% of people perceived as white, a gap of 7.5 points with a range of 4.7 to 10.6. The "
                "check shows the gap exists, not why.",
    "limits": "The arrest model learned what officers decided, not guilt. It must not be used to make decisions about "
              "real people. It only reflects Seattle.",
    "method": "A local language model called Qwen3-4B reads the question and writes the reply. Python works out every "
              "number from the data. The language model was not retrained on the data.",
}

REFUSAL_MESSAGES = {
    "person": "I can't answer questions about a specific person or officer. I can give totals and rates for groups.",
    "outside": "That isn't in this data. I can answer questions about recorded Seattle police stops from 2015 to 2026.",
}

# The plan must follow this shape, which stops the language model from inventing other fields.
PLAN_SHAPE = {
    "type": "object",
    "properties": {
        "action": {"enum": ["stats", "predict", "explain", "refuse"]},
        "measure": {"enum": ["count", "arrest_rate", "frisk_rate", "weapon_rate"]},
        "year_from": {"type": "integer"},
        "year_to": {"type": "integer"},
        "call_words": {"type": "array", "items": {"type": "string"}},
        "precinct": {"enum": PRECINCTS},
        "beat": {"type": "string"},
        "race": {"enum": RACES},
        "gender": {"enum": GENDERS},
        "age_group": {"enum": AGE_GROUPS},
        "outcome": {"enum": OUTCOMES},
        "how_received": {"enum": ["911", "ONVIEW", "TELEPHONE OTHER, NOT 911", "ALARM CALL (NOT POLICE ALARM)"]},
        "weekday": {"enum": WEEKDAYS},
        "month": {"enum": MONTHS},
        "hour": {"type": "integer"},
        "group_by": {"enum": ["year", "precinct", "race", "gender", "age_group", "outcome", "call", "how_received", "hour",
                              "weekday", "month"]},
        "order": {"enum": ["highest", "lowest"]},
        "topic": {"enum": list(PROJECT_FACTS)},
        "refusal": {"enum": list(REFUSAL_MESSAGES)},
    },
    "required": ["action"],
}

# The language model decides these. Python decides everything else in a plan.
LANGUAGE_MODEL_DECIDES = ["action", "group_by", "order", "topic", "refusal"]
FILTER_FIELDS = ["precinct", "beat", "race", "gender", "age_group", "outcome", "how_received", "weekday", "month", "hour", "frisk",
                 "weapon_found", "group_by"]

PLANNING_RULES = f"""You turn a question about Seattle police stops into a JSON plan. Today's year is {CURRENT_YEAR}. Data covers {FIRST_YEAR} to {CURRENT_YEAR}.
Only include the fields the question needs. Never answer the question yourself.

action: "stats" for totals and rates in the data. "predict" for the chance of arrest in a described situation.
"explain" for questions about the data, the AI, its accuracy, fairness or limits. "refuse" otherwise.
measure: count, arrest_rate, frisk_rate or weapon_rate.
call_words: words for the type of call, such as ["robbery"] or ["domestic"].
group_by: use it when the question compares groups or says "by", "which", "most" or "each".
order: use "lowest" only when the question asks for the lowest or least.
refusal: "person" for a named or specific person, officer or ID. "outside" for anything not about these stops, including advice, Jamaica and the law.
A single year sets both year_from and year_to. "Last year" means {CURRENT_YEAR - 1}.

Examples:
Q: How many stops were there in 2024?
{{"action":"stats","measure":"count","year_from":2024,"year_to":2024}}
Q: What share of robbery calls ended in arrest?
{{"action":"stats","measure":"arrest_rate","call_words":["robbery"]}}
Q: Which precinct has the most arrests?
{{"action":"stats","measure":"count","outcome":"Arrest","group_by":"precinct"}}
Q: Arrest rate by perceived race last year
{{"action":"stats","measure":"arrest_rate","year_from":{CURRENT_YEAR - 1},"year_to":{CURRENT_YEAR - 1},"group_by":"race"}}
Q: Which call types have the lowest arrest rate?
{{"action":"stats","measure":"arrest_rate","group_by":"call","order":"lowest"}}
Q: How often are women frisked?
{{"action":"stats","measure":"frisk_rate","gender":"Female"}}
Q: How often was a weapon found in the North precinct?
{{"action":"stats","measure":"weapon_rate","precinct":"North"}}
Q: What is the chance of arrest for a shoplifting call in beat M3 on a Friday at 4pm?
{{"action":"predict","call_words":["shoplifting"],"beat":"M3","weekday":"Friday","hour":16}}
Q: Is the AI fair?
{{"action":"explain","topic":"fairness"}}
Q: How accurate are you?
{{"action":"explain","topic":"accuracy"}}
Q: Where does the data come from?
{{"action":"explain","topic":"data"}}
Q: How many times did officer 7456 arrest someone?
{{"action":"refuse","refusal":"person"}}
Q: How should an officer calm an angry person?
{{"action":"refuse","refusal":"outside"}}
Q: What is the arrest rate in Kingston, Jamaica?
{{"action":"refuse","refusal":"outside"}}"""

WRITING_RULES = """You answer a question using only the facts given. Write one to three plain sentences.
Copy every number exactly as written in the facts. Do not calculate anything new. Do not explain why something happened.
Do not add advice or outside knowledge. If the facts include a warning, keep it."""


class AssistantUnavailable(Exception):
    """The local language model could not be reached."""


class UnanswerableQuestion(Exception):
    """The question was understood, but the data cannot answer it. The message says why."""


@dataclass
class Answer:
    """One finished reply, along with how it was worked out."""
    text: str
    plan: dict = field(default_factory=dict)
    working: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    used_plain_answer: bool = False


class LocalLanguageModel:
    """Wraps the Ollama service, so the rest of the code never deals with its web format."""

    def ask(self, messages, reply_shape="", longest_reply=300):
        """Send messages to the model and return its reply as text."""
        request_body = {"model": LANGUAGE_MODEL, "messages": messages, "stream": False, "think": False, "keep_alive": "15m",
                        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": longest_reply}}
        if reply_shape:
            request_body["format"] = reply_shape
        request = urllib.request.Request(OLLAMA_CHAT_URL, json.dumps(request_body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=LANGUAGE_MODEL_TIMEOUT_SECONDS) as response:
                return json.load(response)["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError) as error:
            raise AssistantUnavailable(f"Could not reach {LANGUAGE_MODEL} through Ollama: {error}") from error


def load_stops():
    """Read the data into one tidy row per stop, leaving out officer and subject identifiers."""
    raw_stops = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    # A stop with two weapons is listed twice, so note the weapon before keeping one row per stop.
    has_weapon = raw_stops["Weapon Type"].notna() & ~raw_stops["Weapon Type"].isin(NO_WEAPON_ENTRIES)
    raw_stops["weapon_found"] = has_weapon.groupby(raw_stops["Terry Stop ID"]).transform("any")
    unique_stops = raw_stops.drop_duplicates("Terry Stop ID").copy()
    occurred_at = pd.to_datetime(unique_stops["Occurred Date"].str.replace("Z", "", regex=False))
    stops = pd.DataFrame({
        "year": occurred_at.dt.year, "month": occurred_at.dt.month_name(), "weekday": occurred_at.dt.day_name(),
        "hour": occurred_at.dt.hour, "call": unique_stops["Initial Call Type"], "how_received": unique_stops["Call Type"],
        "precinct": unique_stops["Precinct"], "sector": unique_stops["Sector"], "beat": unique_stops["Beat"],
        "outcome": unique_stops["Stop Resolution"], "frisk": unique_stops["Frisk Flag"],
        "weapon_found": unique_stops["weapon_found"], "race": unique_stops["Subject Perceived Race"],
        "gender": unique_stops["Subject Perceived Gender"], "age_group": unique_stops["Subject Age Group"]})
    return stops[stops["year"] >= FIRST_YEAR].reset_index(drop=True)


def translate_phrase_to_data_words(phrase):
    """Split a phrase into words, using the data's wording and dropping filler words."""
    words = re.findall(r"[A-Z]+", phrase.upper())
    return [CALL_WORD_SYNONYMS.get(word, word) for word in words if word not in CALL_FILLER_WORDS]


def call_contains_all_words(call, data_words):
    """Check a call type for every word, ignoring bracketed notes such as "(DOES NOT INCLUDE SHOPLIFT)"."""
    main_wording = re.sub(r"\(.*?\)", "", call)
    return bool(data_words) and all(call_states_word(main_wording, word) for word in data_words)


def call_states_word(main_wording, word):
    """Check that the word is present and is not ruled out, as in "NOT DV" or "NO ASLT"."""
    pattern = rf"\b{re.escape(word)}\b"
    is_ruled_out = re.search(rf"\b(?:NOT|NO)\b[^-]*{pattern}", main_wording)
    return bool(re.search(pattern, main_wording)) and not is_ruled_out


def find_matching_calls(call_words, known_calls):
    """Find the data's call types that match any of the asked-for phrases."""
    phrases = [translate_phrase_to_data_words(phrase) for phrase in call_words]
    matches = [call for call in known_calls if call != "-" and any(call_contains_all_words(call, words) for words in phrases)]
    if not matches:
        raise UnanswerableQuestion(f"I couldn't find a call type matching: {', '.join(call_words)}.")
    return matches


def years_named_in_question(question):
    """Read the years from the question's own words. Return nothing if it names no years."""
    lowered = question.lower()
    if "last year" in lowered:
        return CURRENT_YEAR - 1, CURRENT_YEAR - 1
    if "this year" in lowered:
        return CURRENT_YEAR, CURRENT_YEAR
    recent_years = re.search(r"(?:last|past) (\d+) years", lowered)
    if recent_years:
        return CURRENT_YEAR - int(recent_years.group(1)) + 1, CURRENT_YEAR
    named_years = sorted(int(year) for year in re.findall(r"\b((?:19|20)\d\d)\b", lowered))
    if not named_years:
        return ()
    if len(named_years) > 1:
        return named_years[0], named_years[-1]
    return years_around_one_named_year(lowered, named_years[0])


def years_around_one_named_year(lowered_question, year):
    """Work out the range when one year is named, such as "since 2020" or "before 2020"."""
    if re.search(rf"(since|after|from) {year}", lowered_question):
        return year, CURRENT_YEAR
    if re.search(rf"before {year}", lowered_question):
        return FIRST_YEAR, year - 1
    return year, year


def is_follow_up(question):
    """Spot a question that leans on the one before, such as "and in 2023?" or "what about women?"."""
    return bool(re.match(r"\s*(and\b|what about|how about|same\b|now\b|those\b|that\b|for\b|in\b)", question.lower()))


def asks_for_comparison(question):
    """Check that the question really compares groups, so a single figure is not split up unasked."""
    return bool(re.search(COMPARISON_WORDS, question.lower()))


def read_hour(lowered_question):
    """Read a clock time such as "2:30pm" as an hour from 0 to 23. Return -1 if there is none."""
    if "midnight" in lowered_question:
        return 0
    if re.search(r"\bnoon\b", lowered_question):
        return 12
    clock_time = re.search(r"\b(\d{1,2})(?::\d\d)?\s*(am|pm)\b", lowered_question)
    if not clock_time:
        return -1
    hour_on_clock = int(clock_time.group(1)) % 12
    return hour_on_clock + 12 if clock_time.group(2) == "pm" else hour_on_clock


def read_named_value(lowered_question, names):
    """Return the first name found in the question, such as a weekday. Return nothing if none is."""
    for name in names:
        if re.search(rf"\b{name.lower()}s?\b", lowered_question):
            return name
    return ""


def read_filters_from_question(question):
    """Read every detail the question states directly, without using the language model."""
    lowered = question.lower()
    filters = {}
    for field_name, phrases in FILTER_PHRASES.items():
        for pattern, value in phrases:
            if re.search(pattern, lowered):
                filters[field_name] = value
                break
    # "May" and "March" are ordinary words too, so month names count only after "in" or before a year.
    month = next((name for name in MONTHS if re.search(rf"\b(in|during|of) {name.lower()}\b|\b{name.lower()},? \d{{4}}", lowered)), "")
    beat = re.search(r"\bbeat ([a-z]\d)\b", lowered)
    found = {"weekday": read_named_value(lowered, WEEKDAYS), "month": month, "beat": beat.group(1).upper() if beat else ""}
    filters.update({name: value for name, value in found.items() if value})
    if read_hour(lowered) >= 0:
        filters["hour"] = read_hour(lowered)
    if re.search(r"\bfrisk(s|ed)?\b", lowered):
        filters["frisk"] = "Y"
    if re.search(r"\bweapons?\b.*\bfound\b|\bfound\b.*\bweapons?\b|\bhow many weapons\b", lowered):
        filters["weapon_found"] = True
    return filters


def read_group_from_question(question):
    """Read the way of grouping from phrases such as "by race". Return nothing if there is none."""
    for word in re.findall(GROUP_PATTERN, question.lower()):
        if word in GROUP_WORDS:
            return GROUP_WORDS[word]
    return ""


def read_measure_from_question(question, planned_measure):
    """Trust the question's own words for "arrest rate" and "frisk rate", otherwise use the plan."""
    lowered = question.lower()
    for words, measure in (("arrest rate", "arrest_rate"), ("frisk rate", "frisk_rate"), ("weapon rate", "weapon_rate")):
        if words in lowered:
            return measure
    return planned_measure


def clean_call_words(call_words, question):
    """Keep call words the question really used, and join one description's words together."""
    question_words = set(re.findall(r"[A-Z0-9]+", question.upper()))
    kept = []
    for phrase in call_words:
        words = [word for word in re.findall(r"[A-Z0-9]+", phrase.upper()) if word not in NOT_CALL_WORDS]
        if words and set(words) <= question_words:
            kept.append(" ".join(words))
    # "Domestic violence assault" is one description. "Robbery or burglary" is two.
    if len(kept) > 1 and not re.search(r"\b(or|and)\b", question.lower()):
        return [" ".join(kept)]
    return kept


def describe_years(year_from, year_to):
    """Write the period an answer covers, so it can never be mistaken for another period."""
    period = str(year_from) if year_from == year_to else f"{year_from} to {year_to}"
    return f"{period} (data for {CURRENT_YEAR} ends 3 November)" if year_to == CURRENT_YEAR else period


def describe_filter(column, value):
    """Write one filter in plain English."""
    if column == "frisk":
        return "Frisked: yes"
    if column == "weapon_found":
        return "Weapon found: yes"
    return f"{column.replace('_', ' ').capitalize()}: {value}"


def format_share(part, whole):
    """Write a rate along with the counts behind it."""
    return f"{part / whole:.1%} ({part:,} of {whole:,} stops)"


class StopFacts:
    """Works out every number the assistant is allowed to state."""

    # The columns a plan can filter on directly.
    EXACT_FILTERS = ["precinct", "beat", "race", "gender", "age_group", "outcome", "how_received", "weekday", "month", "hour", "frisk",
                     "weapon_found"]
    # Values in a plan that mean "no filter".
    NO_FILTER = {"", None, -1}

    def __init__(self, stops, arrest_model, model_details):
        self.stops = stops
        self.arrest_model = arrest_model
        self.model_details = model_details
        self.known_calls = sorted(stops["call"].unique().tolist())

    def select_stops(self, plan):
        """Apply a plan's filters, and record each one in plain English."""
        chosen = self.stops
        working = []
        year_from = plan.get("year_from") or FIRST_YEAR
        year_to = plan.get("year_to") or CURRENT_YEAR
        chosen = chosen[chosen["year"].between(year_from, year_to)]
        working.append(f"Years: {describe_years(year_from, year_to)}")
        if plan.get("call_words"):
            matches = find_matching_calls(plan["call_words"], self.known_calls)
            chosen = chosen[chosen["call"].isin(matches)]
            working.append("Call types counted: " + "; ".join(matches))
        for column in self.EXACT_FILTERS:
            if plan.get(column, "") not in self.NO_FILTER:
                chosen = chosen[chosen[column] == plan[column]]
                working.append(describe_filter(column, plan[column]))
        return chosen, working

    def measure_flags(self, stops, measure):
        """Return true or false per stop for a rate, leaving out stops where it was not recorded."""
        if measure == "frisk_rate":
            recorded = stops[stops["frisk"].isin(["Y", "N"])]
            return recorded["frisk"].eq("Y")
        if measure == "weapon_rate":
            return stops["weapon_found"]
        return stops["outcome"].eq(ARREST_OUTCOME)

    def describe_measure(self, stops, measure):
        """Write the total or rate for a set of stops."""
        if measure == "count":
            return f"Number of stops: {len(stops):,}"
        flags = self.measure_flags(stops, measure)
        warning = " (few stops, so treat this with care)" if len(flags) < MINIMUM_STOPS_FOR_A_RATE else ""
        return f"{MEASURE_NAMES[measure]}: {format_share(int(flags.sum()), len(flags))}{warning}"

    def rank_groups(self, stops, plan, measure):
        """Work out each group's value and put the groups in the order asked for."""
        group_column = plan["group_by"]
        # Blank labels are record-keeping gaps, so they are left out of comparisons.
        grouped = stops[stops[group_column].ne("-")].groupby(group_column)
        if measure == "count":
            sizes = grouped.size()
            return sizes, sizes
        flags = grouped.apply(lambda group: self.measure_flags(group, measure).agg(["sum", "size"]))
        reliable = flags[flags["size"] >= MINIMUM_STOPS_FOR_A_RATE]
        return reliable["sum"] / reliable["size"], reliable

    def describe_groups(self, stops, plan, measure):
        """Write one line per group, largest first unless the question asked for the lowest."""
        values, details = self.rank_groups(stops, plan, measure)
        if plan["group_by"] in ("year", "hour"):
            ordered = values.sort_index()
        else:
            ordered = values.sort_values(ascending=plan.get("order") == "lowest").head(MOST_ROWS_TO_SHOW)
        if measure == "count":
            return [f"{group}: {int(value):,} stops" for group, value in ordered.items()]
        return [f"{group}: {format_share(int(details.loc[group, 'sum']), int(details.loc[group, 'size']))}" for group in ordered.index]

    def answer_stats(self, plan):
        """Work out a total, a rate or a comparison, and return the fact lines and the working."""
        stops, working = self.select_stops(plan)
        if stops.empty:
            raise UnanswerableQuestion("No stops match that question.")
        measure = plan.get("measure", "count")
        if plan.get("group_by"):
            lines = self.describe_groups(stops, plan, measure)
            if not lines:
                raise UnanswerableQuestion(f"No group has at least {MINIMUM_STOPS_FOR_A_RATE} stops, so I can't compare rates fairly.")
            working.append(f"Compared by {plan['group_by'].replace('_', ' ')}. Rates need at least {MINIMUM_STOPS_FOR_A_RATE} stops. Blank labels left out.")
        else:
            lines = [self.describe_measure(stops, measure)]
        return lines, working

    def build_prediction_rows(self, plan):
        """Take the past stops for this call type and overwrite the details the question gave."""
        matches = find_matching_calls(plan.get("call_words") or [], self.known_calls)
        call_counts = self.stops[self.stops["call"].isin(matches)]["call"].value_counts()
        chosen_call = call_counts.index[0]
        usable = self.stops[(self.stops["call"] == chosen_call) & self.stops["how_received"].ne("-") & self.stops["precinct"].ne("-")]
        rows = usable.rename(columns={"call": "initial_call_type", "how_received": "call_type", "weekday": "day_of_week"}).copy()
        rows["hour_of_day"] = rows["hour"].astype(str)
        return self.apply_given_details(rows, plan), chosen_call

    def apply_given_details(self, rows, plan):
        """Set the details the question gave. Details it did not give keep their real past mix."""
        if plan.get("beat"):
            rows = self.place_rows_in_beat(rows, plan["beat"].upper())
        elif plan.get("precinct"):
            rows = rows[rows["precinct"] == plan["precinct"]]
        replacements = {"call_type": plan.get("how_received"), "day_of_week": plan.get("weekday"), "month": plan.get("month")}
        for column, value in replacements.items():
            if value:
                rows[column] = value
        if plan.get("hour", -1) not in self.NO_FILTER:
            rows["hour_of_day"] = str(plan["hour"])
        if rows.empty:
            raise UnanswerableQuestion("There are no past stops of that kind in that area to base a chance on.")
        return rows

    def place_rows_in_beat(self, rows, beat):
        """Move every row to the asked-for beat, using that beat's real sector and precinct."""
        beat_stops = self.stops[self.stops["beat"] == beat]
        if beat_stops.empty or beat == "-":
            raise UnanswerableQuestion(f"I don't know a Seattle beat called {beat}.")
        rows = rows.copy()
        rows["beat"] = beat
        rows["sector"] = beat_stops["sector"].mode()[0]
        rows["precinct"] = beat_stops["precinct"].mode()[0]
        return rows

    def answer_prediction(self, plan):
        """Ask the arrest model for the chance, averaged over any details the question did not give."""
        if not plan.get("call_words"):
            raise UnanswerableQuestion("To give a chance of arrest, I need to know what the call was about.")
        rows, chosen_call = self.build_prediction_rows(plan)
        uncorrected = self.arrest_model.predict_proba(rows[self.model_details["features"]])[:, 1]
        corrected = expit(logit(uncorrected) + self.model_details["chance_adjustment_from_2024"])
        chance = float(corrected.mean())
        line = (f"Estimated chance of arrest: {chance:.1%}. The average stop in 2024 was {self.model_details['arrest_rate_2024']:.1%}.")
        return [line], [f"Call type used: {chosen_call}", "Details you didn't give use the real past mix of stops for that call."]


# Added to answers from the arrest model, whatever the language model writes.
PREDICTION_NOTE = "The model is weak, and this estimate must not be used to make decisions about real people."

# Added to answers about race or gender, because a rate alone can be misread.
PERCEPTION_NOTE = "Race and gender are as perceived by the officer, and these figures don't show why any differences exist."


class StopQuestionAssistant:
    """Answers one question at a time, and remembers the last plan for follow-up questions."""

    def __init__(self, facts, language_model):
        self.facts = facts
        self.language_model = language_model
        self.last_exchange = []
        self.last_plan = {}

    def plan_question(self, question):
        """Ask the language model for a plan, then make the corrections Python can be sure of."""
        messages = [{"role": "system", "content": PLANNING_RULES}, *self.last_exchange, {"role": "user", "content": question}]
        plan = json.loads(self.language_model.ask(messages, reply_shape=PLAN_SHAPE))
        return self.correct_plan(plan, question)

    def correct_plan(self, plan, question):
        """Replace the plan's details with what the question itself says, keeping earlier details for a follow-up."""
        earlier_plan = self.last_plan if is_follow_up(question) else {}
        corrected = {name: value for name, value in plan.items() if name in LANGUAGE_MODEL_DECIDES}
        corrected.update({name: earlier_plan[name] for name in FILTER_FIELDS if name in earlier_plan})
        corrected.update(read_filters_from_question(question))
        corrected["year_from"], corrected["year_to"] = years_named_in_question(question) or (earlier_plan.get("year_from", 0), earlier_plan.get("year_to", 0))
        corrected["measure"] = read_measure_from_question(question, plan.get("measure", earlier_plan.get("measure", "count")))
        corrected["call_words"] = clean_call_words(plan.get("call_words", []), question) or earlier_plan.get("call_words", [])
        self.correct_grouping(corrected, question, earlier_plan)
        self.remove_filters_that_break_the_measure(corrected)
        return corrected

    def correct_grouping(self, plan, question, earlier_plan):
        """Compare groups only when the question asks to, and use the grouping it names."""
        named_group = read_group_from_question(question)
        wants_comparison = asks_for_comparison(question) or named_group
        chosen_group = named_group or (plan.get("group_by", "") if wants_comparison else earlier_plan.get("group_by", ""))
        plan.pop("group_by", None)
        if chosen_group:
            plan["group_by"] = chosen_group
        if not re.search(r"\b(lowest|least|fewest)\b", question.lower()):
            plan.pop("order", None)

    def remove_filters_that_break_the_measure(self, plan):
        """Remove filters that would make a rate meaningless, such as the arrest rate among arrests."""
        conflicts = {"arrest_rate": "outcome", "frisk_rate": "frisk", "weapon_rate": "weapon_found"}
        plan.pop(conflicts.get(plan["measure"], ""), None)
        # When comparing outcomes, an outcome filter would leave only one group.
        if plan.get("group_by") in plan:
            plan.pop(plan["group_by"], None)

    def gather_facts(self, plan):
        """Run the plan and return the fact lines and the working."""
        if plan["action"] == "stats":
            return self.facts.answer_stats(plan)
        if plan["action"] == "predict":
            return self.facts.answer_prediction(plan)
        raise UnanswerableQuestion(REFUSAL_MESSAGES[plan.get("refusal", "outside")])

    def write_lead_sentence(self, question, fact_lines):
        """Ask the language model to put the main fact into one natural sentence."""
        facts_text = "\n".join(fact_lines)
        messages = [{"role": "system", "content": WRITING_RULES},
                    {"role": "user", "content": f"Facts:\n{facts_text}\n\nQuestion: {question}"}]
        return self.language_model.ask(messages, longest_reply=120)

    def ask(self, question):
        """Answer one question from start to finish."""
        plan = self.plan_question(question)
        if plan["action"] == "explain":
            return self.answer_from_written_notes(plan)
        try:
            fact_lines, working = self.gather_facts(plan)
        except UnanswerableQuestion as reason:
            return Answer(str(reason), plan=plan)
        self.remember_exchange(question, plan)
        return self.finish_answer(question, plan, fact_lines, working)

    def answer_from_written_notes(self, plan):
        """Answer questions about the project with its checked wording, unchanged."""
        topic = plan.get("topic", "method")
        return Answer(PROJECT_FACTS[topic], plan=plan, working=[f"From the project's written notes on: {topic}"])

    def remember_exchange(self, question, plan):
        """Keep the last question and plan, so "what about 2023?" makes sense."""
        self.last_exchange = [{"role": "user", "content": question}, {"role": "assistant", "content": json.dumps(plan)}]
        self.last_plan = plan

    def finish_answer(self, question, plan, fact_lines, working):
        """Write the lead sentence, then add the real figures, the period and any notes."""
        if plan.get("group_by"):
            # Comparison rows always come straight from Python, never through the language model.
            opening, is_trusted = describe_comparison(plan), False
            body = [opening, *fact_lines]
        else:
            opening = self.write_single_fact(question, fact_lines[0])
            is_trusted = opening != fact_lines[0]
            # The counts behind a rate stay visible, even if the sentence leaves them out.
            body = [opening, f"Figures: {fact_lines[0]}"] if is_trusted and find_numbers(fact_lines[0]) - find_numbers(opening) else [opening]
        notes = [describe_what_was_counted(working)] if plan["action"] == "stats" else []
        notes += self.choose_notes(plan)
        text = "\n".join(body) + "".join(f"\n\n{note}" for note in notes)
        return Answer(text, plan=plan, working=working, numbers={"facts": fact_lines}, used_plain_answer=not is_trusted)

    def write_single_fact(self, question, main_fact):
        """Return the language model's sentence if it passes every check, otherwise the plain fact."""
        lead = self.write_lead_sentence(question, [main_fact])
        return lead if lead_sentence_is_trusted(lead, main_fact, question) else main_fact

    def choose_notes(self, plan):
        """Pick the fixed notes this answer must carry."""
        notes = []
        if plan["action"] == "predict":
            notes.append(PREDICTION_NOTE)
        if plan.get("race") or plan.get("gender") or plan.get("group_by") in ("race", "gender"):
            notes.append(PERCEPTION_NOTE)
        return notes


def describe_what_was_counted(working):
    """Say exactly which stops were counted, so a misunderstood question is easy to spot."""
    shown = [line.removeprefix("Years: ") for line in working if not line.startswith(("Call types", "Compared"))]
    calls = [line for line in working if line.startswith("Call types")]
    if calls:
        call_type_count = calls[0].count(";") + 1
        shown.append("1 matching call type" if call_type_count == 1 else f"{call_type_count} matching call types")
    return "Counted: " + "; ".join(shown) + "."


def describe_comparison(plan):
    """Write the heading above a comparison, naming what was measured and any filter used."""
    measure_name = MEASURE_NAMES.get(plan.get("measure"), "Number of stops")
    group_name = plan["group_by"].replace("_", " ")
    outcome = f" that ended in {plan['outcome']}" if plan.get("outcome") else ""
    return f"{measure_name}{outcome} by {group_name}:"


def find_numbers(text):
    """List every number in a piece of text, without commas or a trailing full stop."""
    return {number.replace(",", "").rstrip(".") for number in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def lead_sentence_is_trusted(lead, main_fact, question):
    """Trust the sentence only if it uses the main figure, adds no numbers, and claims no causes."""
    all_fact_numbers = find_numbers(main_fact)
    main_figure = re.search(r"\d[\d,]*(?:\.\d+)?", main_fact).group().replace(",", "").rstrip(".")
    lead_numbers = find_numbers(lead)
    adds_no_numbers = lead_numbers <= all_fact_numbers | find_numbers(question)
    claims_nothing_extra = not any(phrase in lead.lower() for phrase in BANNED_REPLY_PHRASES)
    return main_figure in lead_numbers and adds_no_numbers and claims_nothing_extra


def build_assistant():
    """Load the data and the arrest model, and connect them to the language model."""
    model_details = json.loads(MODEL_DETAILS_PATH.read_text())
    facts = StopFacts(load_stops(), joblib.load(MODEL_PATH), model_details)
    return StopQuestionAssistant(facts, LocalLanguageModel())
