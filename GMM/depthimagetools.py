import numpy as np

class DepthImageTools ():
    @classmethod
    def get_distance_for_pixel (cls, pixel_value):
    	'''
    	Pixel value:
    		0: Am weitesten weg
    		255: Am naehesten
    		Mean: 66.2765903478926
    		Std: 10.309150914167866
    		Min: 1.0
    		Max: 88.0
    	'''
    	
    	'''
    	
    	if isinstance(pixel_value, np.ndarray):
    		sel = pixel_value == 0
    		pixel_value[sel] = min_value
    	else:
    		if pixel_value == 0:
    			pixel_value = min_value
    	
    	# S. 49
    	
    	'''
    	# 255 -> 0.1
    	# 0 -> 300
    	b = 300
    	m = -299.9 / 255
    	d = m*pixel_value + b - (255 - 88)
    	
    	# d = (1 / ((pixel_value / 255) ** 10) * (1 - (300 / 0.1))) - (300 / (0.1 - 300))
    	
    	
    	return d

    @classmethod
    def get_distance (cls, depth_map):
        # 255-0 -> 0-255
        depth_map = 255 - depth_map
        return depth_map

    @classmethod
    def get_alpha (cls, y):
    	alpha = -((30 / (512 / 2)) * (y - (512 / 2)))
    	return alpha

    @classmethod
    def get_height_above_ground (cls, alpha, distance):
    	return np.sin(np.deg2rad(alpha)) * distance
    
    @classmethod
    def remove_empty_space_from_raw (cls, raw_pixel_values):
    	# NICHT INVERTIERT!!
    	raw_pixel_values = raw_pixel_values.astype(np.float)
    	raw_pixel_values *= 255 / 88
    	return raw_pixel_values
    	
    	
    	
    
    @classmethod
    def normalize_value (cls, pixel_value):
    	'''
    	Pixel value:
    		0: Am weitesten weg
    		255: Am naehesten
    		Mean: 66.2765903478926
    		Std: 10.309150914167866
    		Min: 1.0
    		Max: 88.0
    		
    	Return:
    		0: Am weitesten weg
    		1: Am naehesten
    	'''
    	
    	
    	
    	return pixel_value / 255.0
    
    @classmethod
    def normalize_value_reversed (cls, pixel_value):
    	'''
    	Pixel value:
    		0: Am weitesten weg
    		255: Am naehesten
    		Mean: 66.2765903478926
    		Std: 10.309150914167866
    		Min: 1.0
    		Max: 88.0
    		
    	Return:
    		0: Am naehesten
    		1: Am weitesten weg
    	'''
    	return 1 - cls.normalize_value(pixel_value)
    
    @classmethod
    def normalize_value_switch (cls, pixel_value, reverse_values):
    	if reverse_values:
    		pixel_value = cls.normalize_value_reversed(pixel_value)
    	else:
    		pixel_value = cls.normalize_value(pixel_value)
    		
    	return pixel_value
    
    @classmethod
    def get_distance_inverted (cls, pixel_value, reverse_values=False):
    	pixel_value = cls.normalize_value_switch(pixel_value, reverse_values)
    	
    	pixel_value[pixel_value == 0.0] = 1 / 255.0
    	return 1 / pixel_value
    	
    	
