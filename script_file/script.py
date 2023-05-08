import re
import requests
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pyproj
from pyproj import Transformer
from shapely.ops import transform


import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point , Polygon


import matplotlib.pyplot as plt
import folium
import torchvision
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification ,ViTFeatureExtractor
from PIL import Image,ImageOps


from salem import  DataLevels, GoogleVisibleMap, Map
from transformers import pipeline
from tqdm import tqdm


from sklearn.cluster import DBSCAN
import haversine as hs
from haversine import Unit
from scipy.cluster.hierarchy import linkage, dendrogram , fcluster

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform


def extract_land_no(text):
    
    '''extract land_no from text
    return land_no in string format
    if dont have land_no return null
    '''
    try:
        land_no_text = re.findall('[เลข].+' , text)[0]
        land_no = re.split(':[ ]*',land_no_text)[-1]
        return land_no
    except:
        pass

def get_google_sattelite_layer():

    '''This function return non-ovelay google satelite folium tilelayer 
    use to minimize creating tile layer code
    '''
    return folium.TileLayer(
            tiles = 'http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = False,
            show = False)

def imshow(img, text=None):
    
    '''imshow from image which is torch tensor'''
    
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()   

def get_img_salem(location ,r=0.0003,maptype='satellite',crop = 0):
    '''This function return image in PIL format
    Bring API from Salem module
    location = [latitude,longtitude]
    size = image size
    r = width and hieght of picture if r less than 0.0004 , no different
    '''
    
    pic_lat,pic_long = location
    pic_map = GoogleVisibleMap(x=[pic_long-r, pic_long+r], y=[pic_lat-r, pic_lat + r], maptype=maptype)
    
    # Create a salem image data
    img = pic_map.get_vardata()

    #convert to PIL and transform image 

    im = Image.fromarray((img * 256).astype(np.uint8))
    im = im.convert("RGB")
    
    border = (crop, crop, crop, crop) # left, top, right, bottom
    im = ImageOps.crop(im, border)
    return im

def point_buffer(lat: float, lon: float, radius: int,standard_crs = "EPSG:4326"):
    """
    Get the square around a point with a given radius/length in meters.
    """

    # Azimuthal equidistant projection
    aeqd_proj = "+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0"

    transformer = Transformer.from_proj(aeqd_proj.format(lat=lat, lon=lon), standard_crs, always_xy=True)

    buffer = Point(0, 0).buffer(radius,cap_style = 3)

    return transform(transformer.transform, buffer)

def filter_gdf(gdf , detector , candidate_labels = ["building satellite view", "ground satellite view"] , select_list = ['building satellite view'] ,crop = 40 ):
    '''
    filter unwanted row that not a select label which predict my your detector 

    detector is zero-shot image classification
    '''
    label_list = []
    for i in tqdm(range(len(gdf))):
        pic = get_img_salem([gdf.iloc[i]['ltt'],gdf.iloc[i]['lgt']],crop = crop)
        predictions = detector(pic, candidate_labels= candidate_labels)
        label = predictions[0]['label']
        possibility  = predictions[0]['score']

        label_list.append(label) 

    gdf['type'] = label_list
    
    return gdf[gdf['type'].isin(select_list)]


def create_cluster_df(points,labels,point_col = ['ltt','lgt','land_no'] , filter = True):
    '''
    create dataframe from point and label

    point size and index must exactly match with labels
    '''
    #create cluster dataframe
    
    cluster_df = pd.DataFrame(points,columns = point_col)
    cluster_df['cluster'] = labels
    if filter == True:
        cluster_df = cluster_df[cluster_df['cluster'] != -1]
    
    return cluster_df

def point_scan(points  , dbscan , distance ):

    '''
    get label from db scan with point , dbscan object and distance
    '''

    points = points
    
    # Compute the pairwise distances between all points using your custom distance function
    D = pdist(points, metric=distance)

    # Convert the condensed distance matrix to a square distance matrix
    D = squareform(D)
    
    #get labels
    labels = dbscan.fit_predict(D)

    return labels
    


def get_polygon_gdf(cluster_df , lat_long_col = ['ltt','lgt'], cluster_col = 'cluster'):

    '''
    create polygon gdf from latitude longtitude depend on cluster
    '''
    # Extract Polygon from each cluster
    # cluster df need to have ltt lgt and cluster column

    Polygon_gdf = gpd.GeoDataFrame({'geometry':[],cluster_col:[]},crs = "EPSG:4326")

    for cluster in np.unique(cluster_df[cluster_col].values):
        
        focus_df = cluster_df[cluster_df[cluster_col] == cluster]
        
        #point for generate polygon
        
        points = [[focus_df.iloc[i][lat_long_col[0]] ,
                focus_df.iloc[i][lat_long_col[1]]] 
                for i in range(len(focus_df))]
        
        # convert the latitude and longitude points to Point objects
        geometry = [Point(xy) for xy in points]

        # create a GeoDataFrame with the points and labels
        gdf = gpd.GeoDataFrame({'geometry': geometry},crs = "EPSG:4326")

        # select only the points in the target cluster
        cluster_points = gdf['geometry'].tolist()

        # create a polygon around the cluster using the ConvexHull method
        polygon = Polygon([(p.x, p.y) for p in cluster_points]).convex_hull

        # create a GeoDataFrame with the polygon
        gdf = gpd.GeoDataFrame({'geometry': [polygon]})
        gdf[cluster_col] = [cluster]
        
        #collect polygon data
        Polygon_gdf = Polygon_gdf.append(gdf, ignore_index = True)

    return Polygon_gdf

def encode_image(image , feature_extractor , model):

    '''
    encode image with torch model with feature extractor and model
    '''
    
    # Pass the PIL.Image.Image object to the feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    # Use the model to encode the input features and get the logits
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.logits
    return features

def get_embedding_df(cluster_df , feature_extractor , model ,embed_ratio = 0.5 ,crop = 40):

    embedding_output = {'cluster':[]
                    ,'embbeding_vector':[]}
    #embedding and weigth in same cluseter
    for cluster in np.unique(cluster_df['cluster'].values):
        #loop depend on cluster of data
        
        focus_df = cluster_df[cluster_df['cluster'] == cluster]
        #random only 50% of image to encode faster
        index_random = random.sample(range(0, len(focus_df)), int(len(focus_df)*embed_ratio))
        #define to collect embbeding feature
        cluster_features = 0
        # embbeding image in cluster and weigth it
        for i in tqdm(range(len(index_random))):
            
            pic = get_img_salem([focus_df.iloc[i]['ltt'],focus_df.iloc[i]['lgt']],crop = crop)
            image_features = encode_image(pic, feature_extractor , model)
            cluster_features += image_features
        
        #mean the image_features
        cluster_features = cluster_features/len(index_random)
        
        #add to embedding output dict
        embedding_output['cluster'].append(cluster)
        embedding_output['embbeding_vector'].append(cluster_features)
    
    embbeding_df = pd.DataFrame(embedding_output)

    return embbeding_df

def meter_distance(degree_distance):
    ''' 1 degree distance approximate 111,169 meter
    '''
    return 111169*degree_distance

# Extract Polygon from each cluster

def add_polygon_layer(polygon_gdf , polygon_layer , cluster_col = 'cluster' , color = 'red'):

    '''
    Add polygon that already classify to your layer
    '''

    for cluster in np.unique(polygon_gdf[cluster_col].values):
        
        focus_df = polygon_gdf[polygon_gdf[cluster_col] == cluster]
        
        poly = focus_df['geometry'].values[0]

        # extract the vertices from the polygon object
        vertices = list(poly.exterior.coords)

        # create a polygon object for folium
        folium_poly = folium.Polygon(locations=vertices,
                                    popup = folium.Popup(folium.IFrame('<b>CLUSTER</b>: ' + str(cluster))
                                                                        , min_width=100, max_width=100) ,
                                    color= color, 
                                    fill_color= color)

        # add the polygon to the map
        folium_poly.add_to(polygon_layer)





