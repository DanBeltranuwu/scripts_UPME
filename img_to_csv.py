import time
from osgeo import gdal,ogr
import numpy as np
import math 
from PIL import Image, ImageDraw
from shapely.geometry import Point
from shapely.geometry import Polygon
import sys
import json
import numpy as np
import copy
import pyproj
import rasterio as rio
import pandas as pd
import csv

#EXPLIACION RAPIDA DEL SCRIPT: Este escript toma las imagenes correspondientes a las bandas de una imagen satelital sentinel-2, al igual que los poligonos definidos correspondientes a diferentes clases y localidades (aunque se podria manejar solo con clases sin usar localidades) y extrae la informacion de cada pixel a una fila en un archivo csv en el que cada columna se corresponde a una banda, en promedio corriendo en un mac con procesador M1 pro sobre una imagen de 10980x10980 pixeles tomaba al rededor de una hora en terminar. Sin embargo, tambien se incluyeron diversos prints que ilustran el avance del script para darle una idea al usuario de cuanto se puede estar demorando.

#Nombre base de la imagen de la cual se va a extraer la informacion, en este caso son de sentinel, es importante notar que en Sentinel-2 cada banda de la imagen satelital esta dividida en su propia imagen en formato tipo .jp2, por lo que se tiene por ejemplo que la banda 1 de resolucion de 60 metros tiene nombre: T18PXS_20181225T152631_B01_60m.jp2
#Path de donde esta guardada la imagen, en mi caso tenia dentro del directorio del script una carpeta llamada images donde tenia una carpeta por cada imagen
img_path='../imagenes/results/'
wid=8726
hei=11788
#Path completo a la imagen
img_path_name=img_path
#Clases que se van a manejar
classes=['palma','ciudad','agua','vegetacion','suelo_desnudo','cafe_expuesto','cafe_semisombra','cafe_sombra']
#Construccion del path de los poligonos, en mi caso decidi manejar una carpeta dentro del directorio donde tenia el script, dentro de la cual tenia una carpeta por clases y dentro de las mismas por localidad, aunque solo maneje la localidad del cesar por lo que la iteracion por localidades no esta implementada
#TODO: Implementar iteracion por localidades^
base_poly_path='../poligonos/results/'
#Estos poligonos fueron construidos a mano u obtenidos de fuentes oficiales como el mapa de coberturas de 2018 del IDEAM (para la clase palma)
poly_file_type='.shp'
poly_paths = []
#Ejemplo de path: ./polygons/palma_cesar/palma_cesar.shp
for clas in classes:
    path=base_poly_path+clas+'/'+clas+poly_file_type
    poly_paths.append(path)
#Nombre de las bandas a usar, deben estar dentro del path de imagen previamente definido bajo el formato: ./images/<nombre imagen satelital>/<nombre imagen satelital>_<nombre dentro de la lista>m.<formato, puede ser jp2 o tiff>
#Notese que los nombres: corr_10, entr_10 y MOC-1_10, corresponden a mascaras de texturas obtenidas manualmente a traves de la extencion 'r.texture' dentro de la liberira 'GRASS' de QGIS, para estas se uso la banda 8 de 10 metros (no la 8a) con una ventana movil de 5 pixeles de ancho y un desplazamiento de 1 pixel, si se esta ejecutando sobre un mac con chips 'apple silicon' como lo son la serie M (M1,M2,etc.) es probable que al intentar correrlo dandole al boton correspondiente en la interfaz grafica el programa lance un error, sin embargo esto se puede solucionar dando click en la seccion inferior donde dice 'opciones avanzada'>'copiar comando para la consola de python', luego abriendo la consola de python dentro de QGIS y ejecutandoel comando copiado, esto usualmente congela QGIS por al rededor de una hora dependiendo del tamano de la imagen, sin embargo es normal y despues de que transcurra dicho tiempo se obtendran los resultados, los cuales es necesario guardar con la convencion previamente mencionada.
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
#Path y nombre del archivo donde se va a almacenar el .CSV con los datos en forma tabular.
text_prefix='../ascii_files/'
text_suffix='.csv'
out_path = img_path+'cropped/'
#Se guardan las imagenes de mayor resolucion que fueron recortadas por los poligonos (se escogio la banda 2 por ser de 10 metros) en el siguiente arreglo. Se guardan como matrices, las cuales tienen el mismo tamano de las imagenes originales pero en los pixeles que no coincidian con el poligono correspondiente tienen el valor no_data
classes_arr=[]
#Resolucion en metros de la imagen generada tras recortar con los poligonos, dado que el corte se realiza sobre la banda 2 de 10 metros se le asigna valor de 10
res=10
#valor que indica que no hay informacion, se selecciono -100 ya que se identifico que ninguna banda ni mascara de textura tenia este valor en sus rangos
no_data=-100
names_copy=names.copy()
#Todas las otras bandas que dado que no se van a usar como referencia de donde coinciden los poligonos con la imagen, no es necesariorecortarlas, sin embargo si se usa la funcionlaidad gdal.warp para generar nuevas imagenes y que todas queden del mismo tamano (10m por pixel) o en otras palabras escalar aquellas mas pequenas. De igual forma, se les cambia el tipo a .tif ya que se encontro que este era mas facil de manejar con la libreria GDAL.
static_bands=[]
#Sentinela para que en caso de que ya se hayan creado previamente las imagenes no se vuelvan a crear y asi acelerar un poco el proceso al solo leerlas.  
already_loaded=False
for name in names_copy:
    print(f'processing: {name}')
    img = img_path_name+name+get_suffix(name)
    out = out_path+name+'m'+'.tif'
    if not already_loaded:
        # gdal.Warp(out,img,dstNodata = no_data, xRes=res, yRes=res,outputType=6,width=10980,height=10980)
        gdal.Warp(out,img,dstNodata = no_data, outputType=6,width=wid,height=hei)
    loaded_img = gdal.Open(out,gdal.GA_ReadOnly)
    no_data_value=loaded_img.GetRasterBand(1).GetNoDataValue()
    print(f'no_data_value for {name}: {no_data_value}')
    img_arr=loaded_img.ReadAsArray()
    print(f'The size is: {len(img_arr)}x{len(img_arr[0])}')
    static_bands.append(img_arr)

#Corte de las imagenes con los poligonos de clase
for i in range(len(classes)):
    class_name = classes[i]
    poly_path = poly_paths[i]
    name='B2_10'
    print(f'processing: {name} for class:{class_name} ({i+1}/{len(classes)})')
    img = img_path_name+name+get_suffix(name)
    out = out_path+name+'m_'+class_name+'.tif'
    gdal.Warp(out,img,cutlineDSName=poly_path,cropToCutline=False,dstNodata = no_data, outputType=6,width=wid,height=hei)
    loaded_img = gdal.Open(out,gdal.GA_ReadOnly)
    no_data_value=loaded_img.GetRasterBand(1).GetNoDataValue()
    print(f'no_data_value for {name}: {no_data_value}')
    img_arr=loaded_img.ReadAsArray()
    print(f'The size is: {len(img_arr)}x{len(img_arr[0])}')
    classes_arr.append(img_arr)
#Abrir el archivo .CSV para escribir
f = open(text_prefix+'data'+text_suffix, "w")
writer = csv.writer(f)
#Header del archivo con el nombre de las bandas, se hace el split para que solo salga el nombre sin la resolucion, por ejemplo B01 en vez de B01_60
header=[]
for name in names:
    header.append(name.split('_')[0])
header.append('clase')
header.append('lat')
header.append('lon')
writer.writerow(header)
#Escribir los datos de los pixeles seleccionados dentro del archivo, se recorre clase por clase toda la matriz de la imagen recortada (i.e banda 2 recortada para la clase palma) y cuando el pixel tenga informacion diferente a no_data (indicando que pertenece a esa clase) se recorren las bandas en esa posicion para obtener la informacion de cada banda y se escribe esa fila al CSV.
with rio.open('../imagenes/results/B2_10m.tif') as map_layer:
    for k in range(len(classes)):
        class_name = classes[k]
        act_bands = classes_arr[k] 
        act_band = act_bands
        counter=0
        total=len(act_band)*len(act_band[0])
        for i in range(len(act_band)):
            for j in range(len(act_band[0])):
                pixel_vals=[]
                counter+=1
                advance_percent=(counter/total)*100
                print(f'procesando pixel: {counter} de {total} porcentaje: {advance_percent} clase:({k+1}/{len(classes)})')
                if act_band[i][j]!=no_data:
                    for band in static_bands:
                        pixel_vals.append(band[i][j])
                    lat,lon = map_layer.xy(i,j)
                    # pixels2coord= map_layer.xy(i,j)
                    # print(pixels2coord)
                    # print(f'lat:{lat} lon:{lon}')
                    pixel_vals.append(class_name)
                    pixel_vals.append(lat)
                    pixel_vals.append(lon)
                    writer.writerow(pixel_vals)
            
f.close()

print('FIN')
