import yaml
import sys

def check_config():
    print("Checking config.yaml...")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Verify Keys
        required_keys = ["weights", "models"]
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Missing key: {k}")
                
        print(f"Weights Loaded: {config['weights']}")
        print(f"Models Loaded: {config['models']}")
        print("Config Check: PASS")
        
    except Exception as e:
        print(f"Config Check: FAIL - {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_config()
