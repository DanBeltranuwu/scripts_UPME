print('hello world')
# from osgeo.gdalconst import GDT_Float32, GDT_Int16, GDT_Int32, GDT_Byte
# from osgeo import gdal, ogr
import numpy as np
import sys
import copy
import pandas as pd
import csv
from joblib import load, dump, Parallel, delayed
import time
import traceback
import rasterio as rio

#Descripcion general del script: Este script toma las bandas de una imagen satelital sentinel-2 y usando joblib caraga un modelo pre-entrenado con el cual predice pixel a pixel a que clase pertenece para finalmente crear una imagen pintada en la que se pueda ver dicha informacion de manera mas grafica.

#Nombre de las bandas de la imagen. Referirse al script 'ascii.py' que deberia estar junto a este para un mayor detalle de la convencion de nombres 
names=[
    'B2_10',
    'B3_10',
    'B4_10',
    'B5_20',
    'B6_20',
    'B7_20',
    'B8_10',
    'B11_20',
    'B12_20',
    'DEM_12.5',
    'Slopes_12.5',
]

#Funcion para obtener el formato de la imagen, ya que las de sentinel-2 son .jp2 pero las mascaras de textura generadas por qgis tienen formato .tif
def get_suffix(name):
    # return 'm.jp2' if name not in [ 'corr_10', 'entr_10', 'MOC-1_10'] else 'm.tif'
    return 'm.tif'
#Colores definidos para cada clase al momento de colorear la imagen nueva generada, se usa la convencion RGB con valores posibles entre 0 y 255, por ejemplo en el caso de palma (clase 0) tendria valores R:34,G:139,B:34
colors={
    #Palma
    # 0:[34,139,34],
    0:[34,139,34],
    #Ciudad
    1:[212, 51, 15],
    #Agua
    2:[0,191,255],
    #Naturaleza
    3:[0,255,0],
    #Suelo desnudo
    4:[196, 164, 132],
    #Cafe sombra
    5: [39,7,5],
    #Cafe semisombra
    6: [126,68,22],
    #Cafe expuesto
    7: [175,99,36]
    # 5:[0,70,0],
    #Arena
    # 6:[246,215,176]
}
classes=[
    'cafe_expuesto',
    'cafe_semisombra',
    'cafe_sombra',
    'zona_urbanizada',
    'superficie_de_agua',
    'bosque_y_area_semi-natural',
    'suelo_desnudo',
]
wid=8750
hei=11799
#Funcion que devuelve el dataframe que se le va a entregar al modelo, en este caso esta calculando algunos indices que se usaron extra al momento de entrenar el modelo. Basicamente recorre la imagen correspondiente a cada banda/mascara de textura, extrae su informacion, la vectoriza y la vuelve una columna del dataframe de pandas.
def get_model_input(no_data):
    bands=[]
    band_names=[]
    for name in names:
        img = img_path_name+name+get_suffix(name)
        corrected = img_path_name+'corrected_'+name+get_suffix(name)
        print(img)
        print(corrected)
        gdal.Warp(corrected,img,dstNodata = no_data, outputType=6,width=wid,height=hei)
        loaded_img = gdal.Open(corrected,gdal.GA_ReadOnly)
        img_arr=loaded_img.ReadAsArray()
        band_name,res=name.split('_')
        # res=int(res) if res!='12.5' else '12.5'
        # rep=int(res//10) if res!='12.5' else 2
        # # print(rep)
        matrix=np.matrix(img_arr)
        # if rep>1:
        #     matrix=np.repeat(matrix,rep,axis=1)
        #     matrix=np.repeat(matrix,rep,axis=0)
            # print(matrix)
            # matrix=temp
        # print(np.array(matrix.flatten())[0])
        bands.append(np.array(matrix.flatten())[0])
        band_names.append(band_name)
        print(f'{name} og_img_size:{len(img_arr)}x{len(img_arr[0])} new img size:{matrix.shape}')
    bands=np.array(bands)
    bands=np.transpose(bands)
    df = pd.DataFrame(bands,columns=band_names)
    return df

def match_sizes(img):
    gdal.Warp(out,img,dstNodata = no_data, outputType=6,width=wid,height=hei)

def get_ll(ml,i,j):
    return list(ml.xy(i,j))

#Funcion que recibe el dataframe de pandas sobre el cual se quiere predecir, carga el modelo pre-entrenado, obtiene las predicciones del mismo como un vector, luego toma dicho vector y lo transforma en una matriz del tamano de la imagen original. Retorna dicha matriz.
def predict(df):
    # pipe_line = "pipe.joblib"
    # pipe_line = "pipe_MPA.joblib"
    pipe_line = "pipe_MBTI_XGB_1.joblib"
    # pipe_line = "clasificacion_palma_todo"
    # pipe_line = "pipe_MBTI.joblib"
    model = load(pipe_line)
    df_no_lat_lon=df.copy()
    result = model.predict_proba(df_no_lat_lon)
    for i,cl in enumerate(classes):
        df[cl]=result[:,i]
    res=np.argmax(result,axis=1)
    df['clase']=res
    print('done clase')
    # df['clase_label'] = df.apply (lambda row: classes[int(row['clase'])], axis=1)
    # print('done clase label')
    res = np.array(res)
    res=np.reshape(res, (hei, wid))
    latli=[]
    lonli=[]
    total=len(res)*len(res[0])
    counter=0
    # print('crating ili')
    # ili=np.array([item for i,row in enumerate(res) for item in list(np.repeat(i,len(row)))])
    # ili=Parallel(n_jobs=-1)(delayed(get_ili)(item) for i,row in enumerate(res) for item in list(np.repeat(i,len(row))))
    # ili=[*list(np.repeat(i,len(row))) for i,row in enumerate(res)]
    # print('done ili')
    # jli=np.array([item for row in res for item in list(range(len(row))) ])
    # jli=Parallel(n_jobs=-1)(delayed(get_jli)(item) for row in res for item in list(range(len(row))) )
    # jli=[*list(range(len(row))) for row in res]
    # print('done jli')
    # with rio.open('../imagenes/results/B2_10m.tif') as map_layer:
        # coords=np.array(Parallel(n_jobs=-1)(delayed(get_ll)(map_layer,i,j) for i,j in zip(ili,jli)))
        # coords=np.array(Parallel(n_jobs=-1)(list(map_layer.xy(i,j)) for i,j in zip(ili,jli)))
        # coords=np.array([list(map_layer.xy(i,j)) for i,j in zip(ili,jli)])
        # latli=coords[:,0]
        # lonli=coords[:,1]
        # for i in range(len(res)):
        #     for j in range(len(res[0])):
        #         lat,lon = map_layer.xy(i,j)
        #         latli.append(lat)
        #         lonli.append(lon)
        #         counter+=1
        #         advance_percent=(counter/total)*100
        #         print(f'procesando pixel: {counter} de {total} porcentaje: {advance_percent} ')
    # df['x']=ili
    # print('done lat')
    # df['y']=jli
    # print('done lat lon')
    # df.to_feather('classification_results.feather')
    return res

#Funcion que genera el mapa pintado, recibe como input el path a la imagen en color verdadero (TCI) en la mejor resolucion (En este caso era de 10m por pixel), el path de salida de la imagen coloreada generada y el valor de nodata
def paint_map(raster_input, raster_output,nodata=-1):
    try:
        start = time.time()
        in_data,out_data=None,None
        in_data = gdal.Open(raster_input)
        if in_data is None:
            print(f'Unable to open {raster_input}')
            return None

        #Se lee la primera banda de la imagen de inicio, ya que esta solo se va a usar de refencia no es necesarioleer las otras
        band1 = in_data.GetRasterBand(1)
        rows = in_data.RasterYSize
        cols = in_data.RasterXSize
        vals = band1.ReadAsArray(0, 0, cols, rows)
        no_data= band1.GetNoDataValue()

        #Crear la imagen de salida en formato TIF
        driver = in_data.GetDriver()
        driver = gdal.GetDriverByName("GTiff")
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        out_data = driver.Create(raster_output, cols, rows, 3, GDT_Byte, options=options)
        if out_data is None:
            print(f'Could not create output file {raster_output}')
            return None

        #Se obtiene el input del modelo y se predice
        df=get_model_input(no_data)
        print('predicting...')
        classified = predict(df)
        cols = list(colors.keys())
        #Se recorre cada color, por lo que corresponde a cada banda de la imagen nueva (que si integra todas las bandas en si), por eso el 3 para R,G y B. Luego, se itera sobre todas las clases y usando la imagen de base se revisa en que posiciones coincide que la clasificacion equivalen a la clase actual y a esos se les asigna el valor correspondiente en esa banda. Por ejemplo: para palma el color de la banda roja es 34 y para ciudad 212, entonces en la primera iteracion del for externo sobre la primera banda (rojo) de la nueva imagen los valores que tienen 0(palma) en la matriz de preccion se les asigna 34 y a los que tienen 1(ciudad) se les asigna 212 y asi sucesivamente. Cabe notar que se usa la imagen de color original para tener una especie de soporte de forma que si hay algun valor de clase que no se encuentre entre los definidos (por ejemplo 10) el script deje el valor original, esto puede ser util en caso de que se usen otros modelos y solo se quieran pintar ciertas zonas que tengan alta confianza mientras que se desea dejar otras con el color original.
        dem_data=np.array(vals)
        classified[dem_data==no_data]=-1
        classified[dem_data==0]=-1
        classified[dem_data==-100]=-1
        for i in range(3):
            # band1 = in_data.GetRasterBand(i+1)
            # rows = in_data.RasterYSize
            # cols = in_data.RasterXSize
            # vals = band1.ReadAsArray(0, 0, cols, rows)
            # no_data= band1.GetNoDataValue()
            dem_data=np.array(vals)
            # dem_data_2=dem_data.copy()
            # dem_data_2=dem_data_2[dem_data_2!=no_data]
            # max=np.max(dem_data_2)
            # min=np.min(dem_data_2)
            # max2=np.max(dem_data)
            # min2=np.min(dem_data)
            # print(f'minf:{min} max:{max}')
            # print(f'minf:{min2} max:{max2}')
            # mdiff=max-min
            # cols = list(colors.keys())
            for col in cols:
                dem_datas=colors[col]
                val=(dem_datas[i]/255)*1813
                val+=135
                val=dem_datas[i]
                # print(colors)
                # print(dem_datas)
                # print(val)
                # print(col)
                dem_data[classified==col] = val
            out_band = out_data.GetRasterBand(i+1)
            out_band.SetNoDataValue(no_data)
            out_band.WriteArray(dem_data, 0, 0)
            out_band.FlushCache()

        #Se indica que la ubicacion geografica es la misma de la imagen base (TCI en este caso) de forma que al cargarla en programas de manejo de imagenes georeferenciadas como QGIS aparezca donde debe ser
        out_data.SetGeoTransform(in_data.GetGeoTransform())
        out_data.SetProjection(in_data.GetProjection())
        #Se le informa a que color corresponde cada banda (RGB)
        out_data.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        out_data.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        out_data.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        end = time.time()
        print(f'Took:{end - start}s to finish')

        return raster_output

    except Exception as e:
        print(f"Failure to set nodata values on {raster_input}: {repr(e)}")
        traceback.print_exc()
        return None

    finally:
        del in_data
        del out_data

#Metodo main, se puede pasar el nombre de la imagen como parametro para facilidad, por ejemplo pyton3 paint_map.py imagen1.
if __name__=='__main__':
    img_base=sys.argv[1]
    print(f'Revisando: {img_base}')
    img_name=img_base+'_'
    img_path=f'../imagenes/{img_base}/'
    # img_path_name=img_path+img_name
    img_path_name=img_path
    tci='tci_10'
    image_suffix='m.tif'
    img = img_path_name+tci+image_suffix
    out = img_path_name+'painted_tci'+'filtered_v2'+'.tif'
    print(img)
    print(out)
    paint_map(img,out,-1)

# df=get_model_input()
# predict(df)
