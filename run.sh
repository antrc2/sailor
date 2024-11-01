!git clone https://github.com/hiyouga/LLaMA-Factory
!git clone https://github.com/antrc2/sailor
!cp -r sailor/* LLaMA-Factory

eb130e9448967a206d81a626fb70c69af0130801

# !pip install -r LLaMA-Factory/requirements.txt
# !pip install wandb

print("Hello World!")
!CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --num_machines 1 --machine_rank 0 --config_file LLaMA-Factory/fsdp_config.yaml LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train True \
    --evaluation_strategy "steps" \
    --do_eval True \
    --model_name_or_path sail/Sailor-1.8B-Chat \
    --finetuning_type lora \
    --template sailor \
    --dataset_dir LLaMA-Factory/dataset \
    --dataset aya_orca_sft_data \
    --eval_dataset aya_orca_eval_data \
    --cutoff_len 512 \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --eval_steps 10 \
    --preprocessing_num_workers 2 \
    --max_steps -1 \
    --num_train_epochs 5 \
    --gradient_checkpointing True \
    --save_total_limit 15 \
    --save_steps 10 \
    --output_dir checkpoints/Sailor_0.5B_Finetune_LoRA \
    --bf16 True \
    --plot_loss True \
    --report_to wandb \
    --overwrite_output_dir \
    --ignore_pad_token_for_loss True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05





import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

# Đường dẫn tới model đã tải về
model_path = "checkpoints/Sailor_0.5B_Finetune_LoRA"

# Kiểm tra và sử dụng GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải model và tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Chuẩn bị input
input_text = "Cao đẳng FPT Polytechnic là gì?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Cấu hình streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Hàm để chạy model và tạo output
def generate():
    with torch.no_grad():
        model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.8,
            temperature=0.2,
            num_return_sequences=1,
            streamer=streamer
        )

# Chạy generation trong một thread riêng
thread = Thread(target=generate)
thread.start()

# In ra kết quả theo từng token
print(input_text, end="", flush=True)
for token in streamer:
    print(token, end="", flush=True)

thread.join()


!zip -r Sailor.zip checkpoints/Sailor_0.5B_Finetune/checkpoint-350