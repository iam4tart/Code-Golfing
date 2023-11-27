import json

# Function to convert a single record
def convert_record(record):
    system_message = {"role": "system", "content": "You are a professional code golfer who is expert in language syntax and optimization of code in order to reduce the number of lines in the code"}
    user_message = {"role": "user", "content": record["prompt"]}
    assistant_message = {"role": "assistant", "content": record["completion"]}
    return {"messages": [system_message, user_message, assistant_message]}

# Read input file and convert to the desired format
input_filename = "golf_dataset.json"
output_filename = "formatted_dataset.json"

with open(input_filename, "r") as input_file:
    input_data = json.load(input_file)

output_data = [convert_record(record) for record in input_data]

# Write the output to a new file
with open(output_filename, "w") as output_file:
    json.dump(output_data, output_file, indent=2)

print("Conversion completed. Output saved to", output_filename)
