zero_shot = '''<image>Question: {question}
Options:
{options}
Answer:'''

zero_shot_cot = '''<image>Question: {question}
Options:
{options}
Let's think step by step.'''

one_shot = '''<image>Question: {example_question}}
Options:
{example_options}
Answer: {example_answer}
<image>Question: {question}
Options:
{options}'''

one_shot_cot = '''<image>Question: {example_question}}
Options:
{example_options}
{example_rationale}
Answer: {example_answer}
<image>Question: {question}
Options:
{options}'''