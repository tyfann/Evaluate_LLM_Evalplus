from tqdm import tqdm
import transformers, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

start_index = 0
end_index = 164
max_len = 600
STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## problems in evalplus
from evalplus.data import get_human_eval_plus, write_jsonl

plus_problems = get_human_eval_plus()

plus_task_ids = sorted(plus_problems.keys())[start_index: end_index]
plus_prompts = [plus_problems[task_id]['prompt'] for task_id in plus_task_ids]
num_samples = len(plus_prompts)
print("Number of samples: {}".format(num_samples))


def generate_completion_samples_phi1(model, temp, output_file, loop):
    ## defining model
    model = model
    output_file = output_file

    tokenizer = AutoTokenizer.from_pretrained(model)

    model = AutoModelForCausalLM.from_pretrained(model,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)

    model.eval()
    model.to(DEVICE)
    completion_seqs = []
    loops = loop

    generation_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=temp,
        top_p=0.95,
        max_new_tokens=max_len,
    )

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        prompt = plus_prompts[i].replace('    ', '\t')

        prompt_batch_decoder = [prompt]
        ids_batch = [plus_task_ids[i]]

        encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True, max_length=max_len).to(
            DEVICE)
        input_ids = encoding_decoder['input_ids']

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                gen_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    generation_config=generation_config
                )

            gen_tokens = gen_tokens[:, encoding_decoder['input_ids'].shape[-1]:]

            gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):

                    completion_seq = gen_seq
                    for stop_seq in STOP_SEQS:
                        index = completion_seq.find(stop_seq)
                        if index != -1:
                            completion_seq = completion_seq[:index]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = prompt.replace('\t', '    ') + completion_seq

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq
                         }
                    )

    print("Saving results to {}".format(output_file))

    write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    generate_completion_samples_phi1("microsoft/phi-1", 0.2, "phi_1_samples.jsonl", 1)
