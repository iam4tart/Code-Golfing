import json
import openai
# with open('code_golf_dataset_1.json', 'r') as file:
#     training_data = json.load(file)
openai.api_key = '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'

file_name = "training_data.jsonl"


upload_response = openai.File.create(
  file=open(file_name, "rb"),
  purpose='fine-tune'
)
file_id = upload_response.id
upload_response


fine_tune_response = openai.FineTune.create(training_file=file_id)
fine_tune_response


openai.FineTune.create(training_file=file_id, model="davinci")


fine_tune_events = openai.FineTune.list_events(id=fine_tune_response.id)


# Option 1 | if response.fine_tuned_model != null
fine_tuned_model = fine_tune_response.fine_tuned_model
fine_tuned_model

# Option 2 | if response.fine_tuned_model == null
retrieve_response = openai.FineTune.retrieve(fine_tune_response.id)
fine_tuned_model = retrieve_response.fine_tuned_model
fine_tuned_model


new_prompt = ""

answer = openai.Completion.create(
  model=fine_tuned_model,
  prompt=new_prompt,
  max_tokens=100,
  temperature=0
)
answer['choices'][0]['text']