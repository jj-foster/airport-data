from dataclasses import dataclass,field
from locale import normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyproj
from os.path import exists

@dataclass()
class Airport():
    id:int
    name:str
    lat:float
    lon:float
    airport_type:str
    IATA:str
    
    rw_length: list = field(default_factory=list)
    rw_ref_code: list = field(default_factory=list)

def Airport_filter(airports_df,runways_df,airport_types,runways,continents):

    min_len=0 # meters
    m_ft=3.28084
    airport_types=[f"{x}_airport" for x in airport_types]

    ### runway filtering ###
    rw_ref_codes={1:0,2:800,3:1200,4:1800}

    runways_df=runways_df.loc[
        (runways_df['length_ft']>=min_len*m_ft)
        & (runways_df['closed']==0)
        & (runways_df['lighted']==1)
    ]

    ### airport filtering ###

    airports_df=airports_df.loc[
        (airports_df['type'].isin(airport_types))
        & (airports_df['continent'].isin(continents))
        & (airports_df['scheduled_service']=="yes")
        & (airports_df['iata_code']!=np.NaN)
    ]

    airports=[]
    for i,r in airports_df.iterrows():
        airport=Airport(
            id=r['id'],
            name=r['name'],
            lat=r['latitude_deg'],
            lon=r['longitude_deg'],
            airport_type=r['type'].split("_")[0],
            IATA=r['iata_code']
        )
        
        # get runways
        match_rw=runways_df.loc[runways_df['airport_ref']==r['id']]
        for i1,r1 in match_rw.iterrows():
            l=r1['length_ft']*m_ft**-1
            
            airport.rw_length.append(l)
            code=1
            for code_,length in rw_ref_codes.items():
                if l>=length:
                    code=code_
                else:
                    break
            airport.rw_ref_code=code
        
        if airport.rw_length!=[] and len(airport.rw_length)<=runways:
            airports.append(airport)

    return airports

def Map_airports(ax,airports,types,text=False):
    lats=[airport.lat for airport in airports]
    lons=[airport.lon for airport in airports]
    data={
        'rw_ref_codes':[airport.rw_ref_code for airport in airports],
        'type':[airport.airport_type for airport in airports]
    }

    geometry = [Point(xy) for xy in zip(lons,lats)]
    airports_gdf=GeoDataFrame(geometry=geometry,crs='EPSG:4326',data=data)
    airports_gdf=airports_gdf.to_crs("epsg:3395")

    world=gpd.read_file("data/ne_10m_land/ne_10m_land.shp")
    world=world.to_crs("epsg:3395")

    world.plot(ax=ax, color='lightgrey')

    type_colour={"small":'r','medium':'b','large':'g'}
    for airport_type in types:
        x=airports_gdf[airports_gdf['type']==airport_type]
        x.plot(ax=ax,
            marker='o',
            color=type_colour[airport_type],
            markersize=5,
            label=f"{airport_type} airport"
        )

    if text==True:
        for airport in airports:
            ax.text(airport.lon,airport.lat,airport.name)

    ax.set_title(f"European airports - runways > 1800m")
    ax.set_xlim(-1.4e6,3.7e6)
    ax.set_ylim(4.1e6,8.6e6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return plt

def distance_GPS_points_m(lat0,lat1,lon0,lon1):
    dist=np.sqrt((lat1-lat0)**2+(lon1-lon0)**2)

    return dist # meters

def deg2m(lon,lat):

    transformer=pyproj.Transformer.from_crs(crs_from="EPSG:4326",crs_to="EPSG:3395",always_xy=True)
    lon,lat=transformer.transform(lon,lat)

    return lon,lat

def Airport_routes(ax,routes_df,airports,source_airport,dist):
    ####    Plot circle     ####
    
    source_lon,source_lat=deg2m(source_airport.lon,source_airport.lat)

    ax.plot(source_lon,source_lat,marker='x',color='k',markersize=10)
    circle=plt.Circle((source_lon,source_lat),dist*1000,fill=False,ls='--',label=f"{dist}km range")
    ax.add_patch(circle)
    ax.text(source_lon,source_lat, source_airport.name)

    ####    Get routes      ####

    routes_df['route_IATA']=routes_df['source_IATA']+"-"+routes_df["dest_IATA"]
    routes_df=routes_df[routes_df['source_IATA']==source_airport.IATA]

    ####    Filter routes     ####

    routes_filtered=[]
    for index,row in routes_df.iterrows():
        IATA=row['dest_IATA']
        airport=Find_airport(airports,IATA)
        if airport==None:
            continue

        dest_lon,dest_lat=deg2m(airport.lon,airport.lat)
        dest_dist=distance_GPS_points_m(source_lat,dest_lat,source_lon,dest_lon)

        if dest_dist/1000<=dist:   
            #ax.plot((source_lon,dest_lon),(source_lat,dest_lat))
            ax.text(dest_lon,dest_lat,airport.IATA)

            routes_filtered.append({
                "route":f"{source_airport.IATA}-{IATA}",
                "length":int(dest_dist/1000),
                "plane":row['plane_IATA'].split(" ")[0],
                "dest_lat_m":dest_lat,
                "dest_lon_m":dest_lon
            })

    routes_filtered_df=pd.DataFrame.from_dict(routes_filtered)
    max_len=routes_filtered_df["length"].max()
    min_len=routes_filtered_df["length"].min()

    cmap=mpl.cm.plasma
    #norm=mpl.colors.Normalize(vmin=min_len,vmax=max_len)
    norm=mpl.colors.BoundaryNorm(np.linspace(min_len,max_len,10),cmap.N,extend='neither')

    for index,row in routes_filtered_df.iterrows():
        ax.plot((source_lon,row['dest_lon_m']),(source_lat,row['dest_lat_m']),color=cmap(norm(row['length'])))

    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    plt.colorbar(sm,ax=ax,label="Route length (km)")

    #print(routes_filtered_df)

    f=f"results/{source_airport.IATA}_routes.xlsx"
    if exists(f)==False:
        routes_filtered_df.to_excel(f)
    else:
        print(f"{f} exists.")

    return plt

def Find_airport(airports,IATA):
    x=None
    for airport in airports:
        if airport.IATA==IATA:
            x=airport
    
    if x==None:
        return None
    else:
        return x


if __name__=="__main__":
    runways_df=pd.read_csv('data/runways.csv')
    airports_df=pd.read_csv('data/airports.csv')
    routes_df=pd.read_csv("data/routes.csv")

    continents=["EU"]
    airport_types=['small','medium','large']#["small_airport","medium_airport","large_airport"]
    runways=2

    airports=Airport_filter(airports_df,runways_df,airport_types=airport_types,runways=runways,continents=continents)
    # with open("airports.pkl",'wb') as f:
    #     pickle.dump(airports,f)

    # with open("airports.pkl",'rb') as f:
    #     airports=pickle.load(f)

    fig,ax=plt.subplots()
    Map_airports(ax,airports,types=airport_types,text=False)
    
    circle_centre='GVA'
    dist=1500 # km 

    centre_airport=Find_airport(airports,circle_centre)
    assert centre_airport!=None, f"{circle_centre} not found."

    Airport_routes(ax,routes_df,airports,centre_airport,dist=dist)

    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    plt.show()