import sys
import html
from googleapiclient.discovery import build

def main():

    # parsing the command-line arguments
    if len(sys.argv) != 9:
        print("Usage: python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>\n")
        return

    extractionMethod = sys.argv[1]
    developerKey = sys.argv[2]
    cx = sys.argv[3]  # engine key
    openAIKey = sys.argv[4]
    relationIndex = sys.argv[5]
    confidenceThreshold = sys.argv[6]
    query = sys.argv[7]
    k = sys.argv[8]

    # Accessing Google API
    service = build(
        "customsearch", "v1", developerKey=developerKey
    )

    lexicon = dict()
    #query = query.split(" ")
    query_list = query.split(" ")

    while True:
    # our main loop here

        res = (
            service.cse()
            .list(
                q=query,
                cx=cx,
            )
            .execute())

        query = query_list
        # printing the parameters
        print("Parameters:\nClient key: ", developerKey, "\nEngine key: ", cx, "\nQuery: ", ' '.join(query))
        print("\nGoogle Search Results:" + "\n" + "======================")
    
        # if there's less than 10 results (including 0), we simply terminate the program, as per the requirements
        if int(res['searchInformation']['totalResults']) < 10:
            print('Number of results found is less than 10, terminating the program...')
            return

        for i, item in enumerate(res["items"]): # looping the results

            if "fileFormat" in item: # we are skipping if the format is not HTML
                continue

            formattedUrl = item["formattedUrl"]
            htmlTitle = item["htmlTitle"]
            htmlSnippet = item["htmlSnippet"]

            htmlTitle = htmlTitle.split(" ")
            htmlSnippet = htmlSnippet.split(" ")

if __name__ == "__main__":
    main()
