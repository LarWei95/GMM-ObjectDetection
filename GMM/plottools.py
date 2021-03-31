import matplotlib.patches as patches

class PlotTools ():
	@classmethod
	def box_patch_for_detection_box (cls, midx, midy, width, height, linewidth=1, edgecolor="green"):
		startx = midx-width / 2
		starty = midy-height / 2
		
		patch = patches.Rectangle(
			(startx,starty),
			width,
			height,
			linewidth=linewidth,
			edgecolor=edgecolor,
			facecolor='none'
		)
		return patch
	
	@classmethod
	def box_patch_for_detection_series (cls, det_series, linewidth=1, edgecolor="green"):
		midx = det_series["midx"]
		midy = det_series["midy"]
		width = det_series["width"]
		height = det_series["height"]
		
		return cls.box_patch_for_detection_box(
			midx, 
			midy, 
			width, 
			height, 
			linewidth=linewidth, 
			edgecolor=edgecolor
		)