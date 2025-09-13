import json


#to import and start the class
class ConfigManager: 
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = self._load_config()


#Loads the config
    def _load_config(self):
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("Config file not found")
            return {}
        except json.JSONDecodeError:
            print("Error: JSON File is not valid")
            return {}
        

#ito pag kuha ng values
    def get(self, key, default=None):
        return self.config.get(key, default)
    
#ito naman pagseset at pagsasave
    def set(self, key, value):
        self.config[key] = value

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.config, f, indent = 4)