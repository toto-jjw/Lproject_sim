# Lproject_sim/src/config/config_loader.py
import yaml
import os

# Project root: two levels up from this file (src/config/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConfigLoader:
    """
    Loads simulation configuration from YAML file.
    Relative paths in the config are resolved against the project root directory.
    """
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        # If relative, resolve against project root
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        self.config_path = config_path
        self.project_root = PROJECT_ROOT
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML: {exc}")
                return {}

    def resolve_path(self, path: str) -> str:
        """Resolve a path: if relative, make it absolute relative to project root."""
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.project_root, path))

    def get_terrain_config(self):
        return self.config.get("terrain", {})
    
    

    def get_robots_config(self):
        return self.config.get("robots", [])

    def get_environment_config(self):
        return self.config.get("environment", {})
        
    def get_simulation_config(self):
        return self.config.get("simulation", {})

    def get_assets_config(self):
        return self.config.get("assets", {})

    def get_scene_config(self):
        scene_cfg = self.config.get("scene", {})
        # stellar 설정도 scene_config에 포함 (별도 섹션이지만 SceneManager에서 사용)
        stellar_cfg = self.config.get("stellar", {})
        if stellar_cfg:
            scene_cfg["stellar"] = stellar_cfg
        return scene_cfg

    def get_mission_config(self):
        return self.config.get("mission", {})

    def get_config_dir(self):
        return os.path.dirname(os.path.abspath(self.config_path))

    def validate_config(self):
        """
        [New] Basic validation of the loaded configuration.
        """
        required_sections = ["simulation", "assets", "robots", "scene"]
        missing = [s for s in required_sections if s not in self.config]
        
        if missing:
            print(f"[ConfigLoader] Warning: Missing sections: {missing}")
            
        # Validate Robots
        for i, robot in enumerate(self.get_robots_config()):
            if "name" not in robot:
                print(f"[ConfigLoader] Error: Robot #{i} missing 'name'")
            if "prim_path" not in robot:
                print(f"[ConfigLoader] Error: Robot #{i} missing 'prim_path'")
