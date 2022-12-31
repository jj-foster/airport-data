from dataclasses import dataclass,field
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyproj
from os.path import exists
from tqdm import tqdm
import pickle as pkl
from scipy import stats

pd.options.mode.chained_assignment=None

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


    #airports_df.to_excel('airports.xlsx')
    airports_df=airports_df.loc[
        (airports_df['type'].isin(airport_types))
        & (airports_df['continent'].isin(continents))
        & (airports_df['scheduled_service']=="yes")
        & (airports_df['iata_code']!="")
    ]
    #airports_df.to_csv("airports.csv")

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
        
        if airport.rw_length!=[] and (len(airport.rw_length)<=runways or runways==0):
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

    type_marker={"small":'v','medium':'o','large':'s'}
    for airport_type in types:
        x=airports_gdf[airports_gdf['type']==airport_type]
        x.plot(ax=ax,
            marker=type_marker[airport_type],
            color='k',
            markersize=5,
            label=f"{airport_type} airport"
        )

    if text==True:
        for airport in airports:
            ax.text(airport.lon,airport.lat,airport.name)

    return plt

def distance_GPS_points_m(lat0,lat1,lon0,lon1):
    dist=np.sqrt((lat1-lat0)**2+(lon1-lon0)**2)

    return dist # meters

def deg2m(lon,lat):

    transformer=pyproj.Transformer.from_crs(crs_from="EPSG:4326",crs_to="EPSG:3395",always_xy=True)
    lon,lat=transformer.transform(lon,lat)

    return lon,lat

def Airport_routes(routes_df,airports,source_airport,dist:list,ax=None,export=True):
    ####    Plot circle     ####
    
    source_lon,source_lat=deg2m(source_airport.lon,source_airport.lat)

    if ax!=None:
        ax.plot(source_lon,source_lat,marker='x',color='k',markersize=10)
        circle=plt.Circle((source_lon,source_lat),dist[1]*1000,fill=False,ls='--',label=f"{dist[1]}km range")
        ax.add_patch(circle)
        ax.text(source_lon,source_lat, source_airport.name,weight='bold')

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

        if dist[0]<(dest_dist/1000)<=dist[1]:   
            #ax.plot((source_lon,dest_lon),(source_lat,dest_lat))
            if ax!=None:
                ax.text(dest_lon,dest_lat,airport.IATA)

            routes_filtered.append({
                "route":f"{source_airport.IATA}-{IATA}",
                "length":int(dest_dist/1000),
                "plane":row['plane_IATA'].split(" ")[0],
                "airline":row['airline_IATA'],
                "dest_lat_m":dest_lat,
                "dest_lon_m":dest_lon
            })

    if len(routes_filtered)==0:
        raise Exception(f"No routes < {dist}")

    routes_filtered_df=pd.DataFrame.from_dict(routes_filtered)
    max_len=routes_filtered_df["length"].max()
    min_len=routes_filtered_df["length"].min()

    if ax!=None:
        cmap=mpl.cm.plasma
        #norm=mpl.colors.Normalize(vmin=min_len,vmax=max_len)
        norm=mpl.colors.BoundaryNorm(np.linspace(min_len,max_len,10),cmap.N,extend='neither')

        for index,row in routes_filtered_df.iterrows():
            ax.plot((source_lon,row['dest_lon_m']),(source_lat,row['dest_lat_m']),color=cmap(norm(row['length'])),alpha=0.7)

        sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        plt.colorbar(sm,ax=ax,label="Route length (km)")

    # print(routes_filtered_df)
    # print(routes_filtered_df["length"].mean())

    planes=routes_filtered_df["plane"].unique()
    lengths={}
    for plane in planes:
        plane_df=routes_filtered_df[routes_filtered_df["plane"]==plane]
        lengths[plane]=plane_df["length"].mean()

    #print(lengths)

    if export==True:
        f=f"results/{source_airport.IATA}_routes.xlsx"
        if exists(f)==False:
            routes_filtered_df.to_excel(f)
        else:
            print(f"{f} exists.")

    if ax!=None:
        return plt
    else:
        return routes_filtered_df

def Find_airport(airports,IATA):
    x=None
    for airport in airports:
        if airport.IATA==IATA:
            x=airport

    if x==None:
        return None
    else:
        return x

def unique_routes(airports:list,routes_df:pd.DataFrame,dist:list,seats:list):
    # for i,airport in enumerate(tqdm(airports,total=len(airports))):
    #     if i==0:
    #         routes_filtered_df=Airport_routes(routes_df,airports,airport,dist=dist,ax=None,export=False)
    #     else:
    #         try:
    #             routes_new_df=Airport_routes(routes_df,airports,airport,dist=dist,ax=None,export=False)
    #         except Exception:
    #             continue
    #         routes_filtered_df=pd.concat([routes_filtered_df,routes_new_df])

    #     # if i==10:
    #     #     break
    
    # routes_filtered_df.reset_index(inplace=True)
    # routes_filtered_df.drop(["index"],axis=1,inplace=True)

    # with open("results/EU_regional_routes_df.pkl",'wb') as f:
    #     pkl.dump(routes_filtered_df,f)


    with open("results/EU_regional_routes_df.pkl",'rb') as f:
        routes_filtered_df=pkl.load(f)

    aircraft_lookup=pd.read_csv("data/aircraft_seats.csv")
    routes_filtered_df=routes_filtered_df.merge(aircraft_lookup)
    routes_filtered_df.dropna(axis=0,how='any',inplace=True)

    ### Seats per route

    unique_routes=routes_filtered_df["route"].unique()
    route_seats=dict()
    for route in unique_routes:
        seat_sum=routes_filtered_df[routes_filtered_df["route"]==route]["seats"].sum()
        route_seats[route]=seat_sum

    ### Average route length weighted by aircraft seats

    aircraft_data=aircraft_lookup
    counts=[]
    len_avgs=[]
    for i,row in aircraft_data.iterrows():
        routes_plane=routes_filtered_df[routes_filtered_df["plane"]==row["plane"]]
        count=routes_plane.shape[0]
        counts.append(count)

        len_avg=routes_plane["length"].sum()/routes_plane.shape[0]
        len_avgs.append(len_avg)

    aircraft_data["count"]=counts
    aircraft_data["avg_route_len"]=len_avgs
    aircraft_data.dropna(axis=0,how='any',inplace=True)

    aircraft_data=aircraft_data[
        (aircraft_data["seats"]>=seats[0]) &
        (aircraft_data["seats"]<=seats[1])
    ]
    routes_filtered_df=routes_filtered_df[
        (routes_filtered_df["seats"]>=seats[0]) &
        (routes_filtered_df["seats"]<=seats[1])
    ]
    print(routes_filtered_df)

    aircraft_data["seat_sum"]=aircraft_data["count"]*aircraft_data["seats"]
    print(f"Seat sum sum: {aircraft_data['seat_sum'].sum()}")
    
    avg_route_len=(aircraft_data["avg_route_len"]*aircraft_data["count"]).sum()/aircraft_data["count"].sum()
    avg_seats=(aircraft_data["seats"]*aircraft_data["count"]).sum()/aircraft_data["count"].sum()
    print(f"Density weighted average route length (km): {avg_route_len}")
    print(f"Density weighted average seats: {avg_seats}")

    return routes_filtered_df, avg_route_len

def route_len_seat_heatmap(routes_df:pd.DataFrame,avg_route_len:float):
    lengths=routes_df["length"].to_numpy()
    seats=routes_df["seats"].to_numpy()

    ### Gaussian KDE aircraft density

    nbins=100
    k=stats.gaussian_kde([lengths,seats])
    xi,yi=np.mgrid[
        lengths.min():lengths.max():nbins*1j,
        seats.min():seats.max():nbins*1j
    ]
    zi=k(np.vstack([xi.flatten(),yi.flatten()]))    # this bit takes ages

    fig,ax=plt.subplots(figsize=(6,6))
    pc=ax.pcolormesh(xi,yi,zi.reshape(xi.shape),shading='auto',cmap=mpl.cm.jet)
    cbar=plt.colorbar(pc,fraction=0.046,pad=0.04)
    cbar.set_label("Aircraft Kernal Density Estimation")

    ### Average route length
    #x1=[avg_route_len,avg_route_len]
    x1=[930,930]
    y1=[yi.min(),yi.max()]
    ax.plot(x1,y1,color='k',linestyle='--',label="Density Weighted Average Route Length")

    ax.set_xlabel("Route Length (km)")
    ax.set_ylabel("Aircraft PAX")
    ax.set_box_aspect(1)
    ax.legend(facecolor='white',edgecolor='black',fancybox=False,framealpha=1)

    plt.show()

if __name__=="__main__":

    runways_df=pd.read_csv('data/runways.csv')
    airports_df=pd.read_csv('data/airports.csv',na_filter=False)
    airports_df.drop(['home_link','wikipedia_link','keywords'],axis=1,inplace=True)
    routes_df=pd.read_csv("data/routes.csv")

    continents=["EU"]
    airport_types=['small','medium','large']
    runways=0
    source_airport="GVA"
    dist=[500,1500] # km 
    seats=[0,1000]

    airports=Airport_filter(airports_df,runways_df,airport_types=airport_types,runways=runways,continents=continents)
    
    plt.rcParams['font.family']='times new roman'
    #fig,ax=plt.subplots()
    #Map_airports(ax,airports,types=airport_types,text=False)
    
    routes_filtered_df,avg_route_len=unique_routes(airports,routes_df,dist,seats)
    route_len_seat_heatmap(routes_filtered_df,avg_route_len)


    # if source_airport!='':
    #     centre_airport=Find_airport(airports,source_airport)
    #     assert centre_airport!=None, f"{source_airport} not found."

    #     Airport_routes(routes_df,airports,centre_airport,dist=dist,ax=None)
    
    # ax.set_title(f"European airports - runways > 1800m",fontsize=14)

    # ax.set_xlim(-1.4e6,3.7e6)
    # ax.set_ylim(4.1e6,8.6e6)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # ax.set_aspect('equal')
    # ax.legend(loc='upper left')

    #plt.show()