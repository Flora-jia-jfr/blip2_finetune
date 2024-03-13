import pickle
import json

# Specify the path to your pickle file
pickle_file_path = '/home/shared/MCL/vqav2/cached_vqa_demo_data_for_blip/vqa_preprocessed_val.pkl'

# Load the data from the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# # Print the loaded data
# print(data)

# Specify the path for the JSON file where you want to save the data
json_file_path = '/home/shared/MCL/vqav2/cached_vqa_demo_data_for_blip/vqa_preprocessed_val.json'

# Save the loaded data into a JSON file
with open(json_file_path, 'w') as json_file:
    # The default argument for json.dump allows for non-dict types to be handled
    json.dump(data, json_file, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))

print(f"Data saved to JSON file at: {json_file_path}")