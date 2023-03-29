import re
import ast
import sys
import requests
import spacy
import openai
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from googleapiclient.discovery import build
from bs4 import BeautifulSoup


def selectNewQuery(X, seen_queries):
    """Selects a new query by concatenating the subject and object of the pair with the highest confidence value."""
    for key, val in X:
        subj, obj = key
        candidate_query = subj + " " + obj
        if (
            candidate_query.lower() not in seen_queries
        ):  # we make sure that this query has already not been processed
            return candidate_query.lower()
    return str()


def extract_text(url):
    """Connects to URL using requests and extracts plain text using BeautifulSoup"""
    response = requests.get(url, timeout=10)  # we set the timeout as 10 seconds
    soup = BeautifulSoup(response.text, "html.parser")
    plain_text = " ".join(soup.stripped_strings)
    return plain_text


def truncate_text(text):
    """Truncates the text to its 10,000 characters if it's longer than that"""
    if len(text) > 10000:
        print(
            "Trimming webpage content from " + str(len(text)) + " to 10000 characters"
        )
        return text[:10000]
    return text


def printResults(X, selectedRelation):
    """Orders the results first by their confidence values, and prints them one by one. Gets called after displaying 10 URLs or a single iteration is over."""
    sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
    print(
        "================== ALL RELATIONS for "
        + selectedRelation
        + " ("
        + str(len(sorted_X))
        + ") ================="
    )
    for key, value in sorted_X:
        subj, obj = key
        print(
            "Confidence: {:<15} | Subject: {:<25} | Object: {}".format(
                round(value, 8), subj, obj
            )
        )


def get_openai_completion(
    # From the sample script
    prompt,
    model,
    max_tokens,
    temperature=0.2,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    response_text = response["choices"][0]["text"]
    return response_text


def getGptPrompts():
    """Returns the prompts in txt files by reading in them one by order (a total of 4)."""
    prompts = []
    for i in range(1, 5):
        with open("prompt_" + str(i) + ".txt") as file:
            prompts.append(file.read())
    return prompts


def main():
    gpt_prompts = getGptPrompts()

    # parsing the command-line arguments
    if len(sys.argv) != 9:
        print(
            "Usage: python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>\n"
        )
        return

    extractionMethod = sys.argv[1]
    developerKey = sys.argv[2]
    cx = sys.argv[3]  # engine key
    openAIKey = sys.argv[4]
    relationIndex = (
        int(sys.argv[5]) - 1
    )  # we subtract 1 from it as we store the relation_list as a list later on
    confidenceThreshold = float(sys.argv[6])
    query = sys.argv[7]
    k = int(sys.argv[8])

    openai.api_key = openAIKey

    relation_list = ["Schools_Attended", "Work_For", "Live_In", "Top_Member_Employees"]
    selectedRelation = relation_list[
        relationIndex
    ]  # we can directly use the relationIndex here as we subtracted from 1

    # Accessing Google API
    service = build("customsearch", "v1", developerKey=developerKey)

    X = dict()  # this is the dictionary for storing all results
    url_counter = 0  # we keep track of the index of the URL, which cannot exceed 10 for a single iteration
    iteration = 0  # number of total iterations

    seen_urls = (
        set()
    )  # we have the seen urls as a set so that we don't process the same URL twice in SpanBERT
    seen_queries = {
        query.lower()
    }  # we convert all queries to lowercase so that we can avoid producing a query that has the same content with a different case

    entities_of_interest_dict = {
        "Schools_Attended": ["PERSON", "ORGANIZATION"],
        "Work_For": ["PERSON", "ORGANIZATION"],
        "Live_In": ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"],
        "Top_Member_Employees": ["ORGANIZATION", "PERSON"],
    }

    entities_of_interest = entities_of_interest_dict[selectedRelation]
    nlp = spacy.load("en_core_web_lg")  # loading in
    spanbert = SpanBERT("./pretrained_spanbert")

    while True:
        # our main loop here

        res = (
            service.cse()
            .list(
                q=query,
                cx=cx,
            )
            .execute()
        )

        # printing the parameters
        print(
            "Parameters:\nClient key: ",
            developerKey,
            "\nEngine key: ",
            cx,
            "\nOpenAI key: ",
            openAIKey,
            "\nMethod: ",
            extractionMethod,
            "\nRelation: ",
            selectedRelation,
            "\nThreshold: ",
            confidenceThreshold,
            "\nQuery: ",
            query,
            "\n# of Tuples: ",
            k,
        )
        print("\nGoogle Search Results:" + "\n" + "======================")

        # if there's less than 10 results (including 0), we simply terminate the program, as per the requirements
        if int(res["searchInformation"]["totalResults"]) < 10:
            print("Number of results found is less than 10, terminating the program...")
            return

        print(
            "=========== Iteration: "
            + str(iteration)
            + " - Query: "
            + query
            + " ==========="
        )

        for i, item in enumerate(res["items"]):  # looping the results
            if "fileFormat" in item:  # we are skipping if the format is not HTML
                continue

            formattedUrl = item["link"]
            url_counter += 1
            print("\n\n\nURL ( " + str(url_counter) + " / 10): " + formattedUrl)

            if (
                formattedUrl not in seen_urls
            ):  # we only process if the url has already not been processed in our program
                try:
                    print("Fetching text from URL...")
                    text = extract_text(formattedUrl)
                except requests.exceptions.RequestException:
                    print("Webpage retrival failed, skipping...")
                    continue

                text = truncate_text(text)
                print("Webpage length (num characters): " + str(len(text)))
                # seen_urls.add(formattedUrl)  # adding it to the seenURLs
            else:
                print("URL already processed; skipping... ")
                continue

            print("Annotating the webpage using spacy...")
            doc = nlp(text)
            total_sentences = (
                0  # keeping track of all sentences where we can extract relations from
            )
            extracted_count = 0  # counter to keep track of how many new relations we extracted in this URL (always less than overall_count naturally)
            overall_count = 0  # counter to keep track of how many relations we extracted in this URL
            seen_sentences = set()

            # total_sentences = len(doc.sents)

            for sentence in doc.sents:
                total_sentences += 1
                gpt_sentence = (
                    str()
                )  # we will use this variable to see if we should feed this sentence to GPT or not, based on the results we get from spacy

                candidate_pairs = []
                sentence_entity_pairs = create_entity_pairs(
                    sentence, entities_of_interest
                )
                for ep in sentence_entity_pairs:
                    if (
                        selectedRelation == "Schools_Attended"
                        or selectedRelation == "Work_For"
                    ):
                        if (ep[1][1], ep[2][1]) == ("PERSON", "ORGANIZATION"):
                            candidate_pairs.append(
                                {"tokens": ep[0], "subj": ep[1], "obj": ep[2]}
                            )

                            gpt_sentence = str(
                                sentence
                            )
                            # seen_sentences.add(str(sentence))
                            # we make gpt_sentence equal to the current sentence only if the corresponding tags are matched for the given relation, so we don't have
                            # to feed all sentences for financial reasons.
                        elif (ep[2][1], ep[1][1]) == ("PERSON", "ORGANIZATION"):
                            candidate_pairs.append(
                                {"tokens": ep[0], "subj": ep[2], "obj": ep[1]}
                            )
                            gpt_sentence = str(
                                sentence
                            )
                    elif selectedRelation == "Top_Member_Employees":
                        if (ep[1][1], ep[2][1]) == ("ORGANIZATION", "PERSON"):
                            candidate_pairs.append(
                                {"tokens": ep[0], "subj": ep[1], "obj": ep[2]}
                            )
                            gpt_sentence = str(sentence)

                        elif (ep[2][1], ep[1][1]) == ("ORGANIZATION", "PERSON"):
                            candidate_pairs.append(
                                {"tokens": ep[0], "subj": ep[2], "obj": ep[1]}
                            )
                            gpt_sentence = str(sentence)

                    elif selectedRelation == "Live_In":
                        if ep[1][1] == "PERSON" and ep[2][1] in [
                            "LOCATION",
                            "CITY",
                            "STATE_OR_PROVINCE",
                            "COUNTRY",
                        ]:
                            candidate_pairs.append(
                                {"tokens": ep[0], "subj": ep[1], "obj": ep[2]}
                            )
                            gpt_sentence = str(sentence)

                        elif ep[2][1] == "PERSON" and ep[1][1] in [
                            "LOCATION",
                            "CITY",
                            "STATE_OR_PROVINCE",
                            "COUNTRY",
                        ]:
                            candidate_pairs.append(
                                {"tokens": ep[0], "subj": ep[2], "obj": ep[1]}
                            )
                            gpt_sentence = str(sentence)

                # Classify Relations for all Candidate Entity Pairs using SpanBERT
                candidate_pairs = [
                    p
                    for p in candidate_pairs
                    if not p["subj"][1] in ["DATE", "LOCATION"]
                ]  # ignore subject entities with date/location type

                if len(candidate_pairs) == 0:
                    continue

                if extractionMethod == "-spanbert":
                    relation_preds = spanbert.predict(
                        candidate_pairs
                    )  # get predictions: list of (relation, confidence) pairs

                    for ex, pred in list(zip(candidate_pairs, relation_preds)):
                        if (
                            (
                                selectedRelation == "Schools_Attended"
                                and pred[0] == "per:schools_attended"
                            )
                            or (
                                selectedRelation == "Work_For"
                                and pred[0] == "per:employee_of"
                            )
                            or (
                                selectedRelation == "Live_In"
                                and pred[0] == "per:cities_of_residence"
                            )
                            or (
                                selectedRelation == "Top_Member_Employees"
                                and pred[0] == "org:top_members/employees"
                            )
                        ):
                            if pred[1] >= confidenceThreshold:
                                seen_sentences.add(sentence)
                                print("=== Extracted Relation ===")
                                print(
                                    "Input tokens: {}".format(
                                        [token.text for token in sentence]
                                    )
                                )
                                print(
                                    "\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(
                                        ex["subj"][0], ex["obj"][0], pred[0], pred[1]
                                    )
                                )
                                pred_str = "{:.7f}".format(pred[1])
                                pred_float = float(
                                    pred_str
                                )  # we convert the prediction into a float here
                                overall_count += 1  # incrementing the overall count
                                if (ex["subj"][0], ex["obj"][0]) not in X.keys():
                                    X[(ex["subj"][0], ex["obj"][0])] = pred_float
                                    extracted_count += 1
                                    print(
                                        "Adding to the set of extracted relations... "
                                    )
                                else:
                                    currentConfidence = pred_float
                                    existingConfidence = X[
                                        (ex["subj"][0], ex["obj"][0])
                                    ]

                                    if currentConfidence < existingConfidence:
                                        print(
                                            "Duplicate with a lower confidence, skipping..."
                                        )
                                    elif currentConfidence > existingConfidence:
                                        print(
                                            "Duplicate with a higher confidence, replacing..."
                                        )
                                        X[
                                            (ex["subj"][0], ex["obj"][0])
                                        ] = currentConfidence
                                        extracted_count += 1  # incrementing the extracted count as we have a higher confidence and we replace
                                    else:
                                        print(
                                            "Duplicate with a same confidence, skipping..."
                                        )
                                print("==================")

                    print("Processed " + str(total_sentences) + " sentences thus far.")
                elif extractionMethod == "-gpt3":
                    if gpt_sentence == "":
                        continue
                    if total_sentences != 0 and total_sentences % 5 == 0:
                        print(
                            "Processed "
                            + str(total_sentences)
                            + " sentences thus far..."
                        )
                    prompt_text = gpt_prompts[relationIndex]
                    prompt_text += gpt_sentence
                    model = "text-davinci-003"
                    max_tokens = 100
                    temperature = 0.2
                    top_p = 1
                    frequency_penalty = 0
                    presence_penalty = 0
                    response_text = get_openai_completion(
                        prompt_text,
                        model,
                        max_tokens,
                        temperature,
                        top_p,
                        frequency_penalty,
                        presence_penalty,
                    )

                    if (
                        "no results" in response_text.lower()
                    ):  # we expect GPT3 to output no results if it couldn't find any results
                        continue
                    else:
                        try:
                            pairs = ast.literal_eval(
                                response_text
                            )
                            seen_sentences.add(sentence)
                            # the format GPT produces will be like [(s1, o1)] per our request. So we use literal_eval function to have 'pairs' as list of tuples.
                            for pair in pairs:
                                print("=== Extracted Relation ===")
                                print("Sentence: ", sentence)
                                print(
                                    "Subject: "
                                    + pair[0]
                                    + " ; Object: "
                                    + pair[1]
                                    + " ; "
                                )
                                if pair not in X:
                                    X[pair] = 1.0
                                    extracted_count += 1
                                    print(
                                        "Adding to the set of extracted relations... "
                                    )
                                else:
                                    print("Duplicate, skipping...")
                                overall_count += 1
                            print("============")
                        except:  # ideally comprehensive except statement like this is not what we want, but here we go with this option as GPT3 can produce anything. so if any exceptions occur while we're converting the expression, we should pass the current relation
                            pass

            # Printing the stats for the current URL before proceeding to the next one

            print(
                "Extracted annotations for "
                + str(len(seen_sentences))
                + " out of total "
                + str(total_sentences)
            )
            print(
                "Relations extracted from this website: "
                + str(extracted_count)
                + " (Overall: "
                + str(overall_count)
                + ") "
            )

        iteration += 1  # incrementing the iteration
        printResults(
            X, selectedRelation
        )  # printing the results, regardless of we reached the desired tuples or not
        if len(X) >= k:
            break  # we quit the program if the desired tuples are reached
        else:
            url_counter = 0  # resetting the counter before proceeding to the next URL
            sorted_X = sorted(
                X.items(), key=lambda item: item[1], reverse=True
            )  # we sort the X before passing it into selecting new query function, as we want to select the highest confidence query terms to build up the new query
            query = selectNewQuery(sorted_X, seen_queries)
            if query == str():
                break
            seen_queries.add(
                query
            )  # adding to the seen queries so we don't run the same query twice


if __name__ == "__main__":
    main()
