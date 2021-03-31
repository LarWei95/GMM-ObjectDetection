from datatools import DataConverter
from depthimagetools import DepthImageTools

def create_gmm_dataframe ():
	joined_df = DataConverter.get_gmm_dataframe(DepthImageTools.get_distance_inverted, tuple([True]))
	return joined_df

def default():
	gmm_df = create_gmm_dataframe()
	gmm_df.to_csv("gmm_data.csv")

if __name__ == '__main__':
	default()
