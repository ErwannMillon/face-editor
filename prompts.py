import random
class PromptSet:
    def __init__(self, pos, neg, config=None):
        self.positive = pos
        self.negative = neg
        self.config = config
example_prompts = (
    PromptSet("a picture of a woman with light blonde hair", "a picture of a person with dark hair | a picture of a person with brown hair"),
    PromptSet("A picture of a woman with very thick eyebrows", "a picture of a person with very thin eyebrows | a picture of a person with no eyebrows"),
    PromptSet("A picture of a woman wearing bright red lipstick", "a picture of a person wearing no lipstick | a picture of a person wearing dark lipstick"),
    PromptSet("A picture of a beautiful chinese woman | a picture of a Japanese woman | a picture of an Asian woman", "a picture of a white woman | a picture of an Indian woman | a picture of a black woman"),
    PromptSet("A picture of a handsome man | a picture of a masculine man", "a picture of a woman | a picture of a feminine person"),
    PromptSet("A picture of a woman with a very big nose", "a picture of a person with a small nose | a picture of a person with a normal nose"),
)
def get_random_prompts():
    prompt = random.choice(example_prompts)
    return prompt.positive, prompt.negative