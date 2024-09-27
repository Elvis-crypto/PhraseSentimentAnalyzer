# phrase_config.py

import yaml

class Config:
    def __init__(self, config_file='../config.yaml'):
        self.phrases = []
        self.reddit_config = {}
        self.load_config(config_file)

    def load_config(self, config_file):
        """
        Load configuration from a YAML file.
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.phrases = config.get('phrases', [])
        self.reddit_config = config.get('reddit', {})

# Test the module
if __name__ == "__main__":
    config = Config()
    print("Phrases to monitor:")
    for phrase in config.phrases:
        print(f"- {phrase}")
    print("\nReddit Configuration:")
    print(config.reddit_config)
