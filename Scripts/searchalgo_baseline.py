from sentence_transformers import SentenceTransformer, util
import pandas as pd
import csv


# Define filters (keywords to search for)
def get_filters_from_user():
    print("Enter the filters (keywords) separated by commas:")
    user_input = input()
    filters = [keyword.strip() for keyword in user_input.split(",")]
    return filters

filters = get_filters_from_user()

# Replace the path to the input data file
file_path = '../Data/program_data_extended.csv'

# Read the CSV file
data = pd.read_csv(file_path)


# Store the Title and description in a dictionary
title_description_dict = dict(zip(data['Title'], data['Description']))
print("Description is:")
print(data['Description'].head(5))

# Print the first element in the dictionary
first_key = next(iter(title_description_dict))
#print("1st element in dictionary", first_key, title_description_dict[first_key])

# Initialize dictionary for Title scores and matched filters
title_scores = {title: 0 for title in title_description_dict.keys()}
matched_filters = {title: [] for title in title_description_dict.keys()}


# Initialize Sentence Transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode the filters
filter_embeddings = model.encode(filters, convert_to_tensor=True)

# Loop through each Title and description in the dictionary
for title, description in title_description_dict.items():
    try:
        # Combine all text elements (in this case, just the description)
        all_text_elements = [description]

        # Update the score for each occurrence of a filter word
        for text in all_text_elements:
 
            # Encode the text
            text_embedding = model.encode(text, convert_to_tensor=True)
            # Compute cosine similarity between the text and filter embeddings
            cosine_scores = util.pytorch_cos_sim(text_embedding, filter_embeddings)
            max_score = cosine_scores.max().item()
            if max_score > 0.5:  # Threshold for considering a match
                title_scores[title] += max_score
                matched_filters[title].append((filters[cosine_scores.argmax().item()], text))

    except Exception as e:
        print(f"Failed to process data for {title}: {e}")

# Sort Titles based on scores in descending order
sorted_titles = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)

# Print sorted Titles along with match counts and matched content if score > 0.5
for title, score in sorted_titles:
    if score > 0.5:
        print(f"Title: {title}, Score: {score}")
        with pd.ExcelWriter('../Data/filtered_titles_largerdataset.xlsx', engine='xlsxwriter') as writer:
            rows = []
            for title, score in sorted_titles:
                if score > 0.5:
                    for filter_word, _ in matched_filters[title]:
                        rows.append({'Filter': filter_word, 'Title': title, 'Score': score})
            df = pd.DataFrame(rows, columns=['Filter', 'Title', 'Score'])
            df.to_excel(writer, index=False)   
