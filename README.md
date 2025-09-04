# PPO AI Safety Training

This project implements Proximal Policy Optimization (PPO) training to improve AI safety by training language models to refuse harmful requests while maintaining helpfulness for benign queries.

## Overview

The system trains a language model using reinforcement learning to:
- **Refuse harmful requests** (e.g., hacking scripts, malware creation)
- **Maintain helpfulness** for benign queries (e.g., educational content)

It uses a reward system that encourages appropriate refusal behavior for disallowed prompts while maintaining normal responses for safe prompts.

## Features

- **Flexible Training Backend**: Automatically detects and uses TRL (Transformers Reinforcement Learning) if available, falls back to a simple training loop
- **Hybrid Reward System**: Combines rule-based heuristics with optional reward model scoring
- **Safety-Focused**: Specifically designed to improve AI safety through refusal training
- **Comprehensive Evaluation**: Compares base model vs trained model responses
- **Results Tracking**: Saves training logs and generates comparison tables

## Requirements

### Core Dependencies
```bash
pip install torch transformers pandas numpy
```

### Optional (Recommended)
```bash
pip install trl  # For advanced PPO training
```

## Project Structure

```
├── main.py              # Main training script
├── results.md           # Generated evaluation results (after training)
├── ppo_training_results_*.csv  # Training logs with timestamps
└── ppo_trained_model/   # Saved trained model directory
```

## Usage

### Basic Training
```bash
python main.py
```

### Expected Output
The script will:
1. Load GPT-2 base model and tokenizer
2. Attempt to load a reward model (if available)
3. Train for 200 epochs using PPO or simple training
4. Save training logs and the final model
5. Generate evaluation comparisons

## Configuration

### Training Parameters
- **Model**: GPT-2 (configurable via `policy_model_name`)
- **Learning Rate**: 1.41e-5
- **Training Epochs**: 200
- **Generation Length**: 50 tokens max

### Prompt Categories

**Benign Prompts** (should get helpful responses):
- Educational content (photosynthesis, geography)
- Science explanations
- General knowledge queries

**Disallowed Prompts** (should get refusals):
- Hacking instructions
- Malware creation guides
- Breaking and entering advice

### Reward System

The reward function operates on a -1 to 1 scale:

| Prompt Type | Response Type | Reward |
|-------------|---------------|--------|
| Disallowed  | Refusal       | +1.0   |
| Disallowed  | Compliance    | -1.0   |
| Benign      | Helpful       | +0.5   |
| Benign      | Refusal       | -1.0   |

## Refusal Detection

The system identifies refusal responses by looking for phrases like:
- "I'm sorry"
- "I can't" / "I cannot" 
- "I'm unable"
- "As an AI"
- "This request is inappropriate"

## Training Modes

### 1. TRL PPO Mode (Preferred)
Uses the TRL library's PPOTrainer for sophisticated reinforcement learning:
```python
ppo_trainer = PPOTrainer(ppo_config, policy_model, None, tokenizer)
```

### 2. Simple Training Mode (Fallback)
Uses a basic training loop with reward-scaled loss:
```python
scaled_loss = loss * (1.0 - reward)
```

## Output Files

### Training Results CSV
Contains epoch-by-epoch training data:
```csv
epoch,prompt,response,reward
1,"Explain photosynthesis...","Photosynthesis is...",0.5
```

### Evaluation Results (results.md)
Markdown table comparing base model vs trained model responses:
```markdown
| Prompt | Base Model Output | PPO Model Output |
|---|---|---|
| Explain photosynthesis... | Photosynthesis is... | Photosynthesis is... |
```

## Customization

### Adding New Prompts
Modify the prompt lists in the script:
```python
benign_prompts = [
    "Your new benign prompt here",
    # ... existing prompts
]

disallowed_prompts = [
    "Your new disallowed prompt here",
    # ... existing prompts
]
```

### Adjusting Refusal Detection
Modify the `refusal_phrases` list in the `is_refusal()` function:
```python
refusal_phrases = [
    "I'm sorry",
    "Your custom refusal phrase",
    # ... other phrases
]
```

### Custom Reward Model
To use your own reward model:
```python
reward_model_path = "/path/to/your/reward/model"
```

## Troubleshooting

### Common Issues

**TRL Import Error**: 
- Install TRL: `pip install trl`
- Or continue with simple training (automatic fallback)

**CUDA Out of Memory**:
- Reduce batch size in PPOConfig
- Use smaller model (e.g., "distilgpt2")
- Force CPU usage: Remove CUDA device placement code

**Reward Model Not Found**:
- Update `reward_model_path` to correct location
- Training continues with rule-based rewards only

### Performance Tips

- **GPU Usage**: Automatically uses CUDA if available
- **Memory Optimization**: Uses gradient clipping and small batch sizes
- **Error Handling**: Robust fallbacks for component failures

## Technical Details

### PPO Implementation
The training uses Proximal Policy Optimization principles:
- Policy gradient methods with clipped surrogate objective
- Reward-based loss scaling
- Gradient clipping for stability

### Safety Measures
- **Explicit refusal training** for harmful content
- **Maintained helpfulness** for benign requests
- **Rule-based safety** as fallback for reward model failures

## Contributing

To extend this project:
1. Add new prompt categories in the prompt lists
2. Enhance the refusal detection logic
3. Implement additional reward model architectures
4. Add more sophisticated evaluation metrics

## License

This project is for educational and research purposes in AI safety.

---

**Note**: This implementation is designed for research and educational purposes. For production AI safety applications, consider more robust training methodologies and comprehensive safety evaluations.
