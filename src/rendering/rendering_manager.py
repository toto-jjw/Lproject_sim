# Lproject_sim/src/rendering/rendering_manager.py
import carb

class RenderingManager:
    """
    Manages rendering settings and post-processing effects.
    """
    def __init__(self):
        self.settings = carb.settings.get_settings()
        
    def enable_lens_flare(self, enable: bool = True):
        """Enables/Disables lens flare effect."""
        self.settings.set("/rtx/post/lensFlares/enabled", enable)
        
    def set_lens_flare_params(self, scale: float = 1.0, blades: int = 9, fstop: float = 2.8, focal_length: float = 12.0):
        """Sets lens flare parameters."""
        self.settings.set("/rtx/post/lensFlares/flareScale", scale)
        self.settings.set("/rtx/post/lensFlares/blades", blades)
        self.settings.set("/rtx/post/lensFlares/fNumber", fstop)
        self.settings.set("/rtx/post/lensFlares/focalLength", focal_length)
        
    def enable_motion_blur(self, enable: bool = True):
        """Enables/Disables motion blur."""
        self.settings.set("/rtx/post/motionblur/enabled", enable)
        
    def enable_dlss(self, enable: bool = True):
        """Enables/Disables DLSS if available."""
        self.settings.set("/rtx/post/dlss/enabled", enable)

