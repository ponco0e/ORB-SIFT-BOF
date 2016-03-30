/*
 * Generador de categorias para algoritmo bag of words.
 * Se introducen como parametros la direccion de un dataset de imagenes
 * y el numero de imagenes a procesar. Se obtienen descriptores SIFT de
 * cada imagen, se cuantizan por k-means, y se guardan los resultados
 *
 * Alfonso R.O.
 */



#include <iostream>
#include <vector>
#include <algorithm>  //para la validacion 10-fold cruzada
#include "opencv2/opencv.hpp"
#include "bofgenerator.h"



using namespace cv;

float validate(int argc, char** argv){
    Mat image; //declarar imagen de opencv
    int algotype = atoi(argv[argc-1]);
    int nwords = atoi(argv[argc-2]);
    int nimg = atoi(argv[argc-3]);
    int nclasses = argc-4;
    int chunksize = (nclasses*nimg)/10; //numero de clases * numero de imagenes entre 10
    vector<String> imagepaths; //todas las paths a imagenes
    vector<int> imageclasses; //las clases de cada imagen
    vector<int> permindex; //indices de los vectores para permutacion

    char numbers[4];
    for(int p=1; p<=nclasses; p++){ //por cada dataset
        String imagedir(argv[p]); //cargar su path a una string
        for(int i=0; i<nimg; i++){ //cargar imagenes de la 1 a la number_of_images
            sprintf(numbers, "%04d", i+1); //convertir el indice en cadena digitos 0001-9999
            String imagepath = imagedir + "/image_" + numbers + ".jpg"; //crear la direccion de la imagen
            cout << "DBG: " << imagepath << endl;
            image = imread( imagepath, CV_LOAD_IMAGE_GRAYSCALE ); //abrir imagen como escala de grises
            if ( !image.data )
            {
                cout<<"ERROR: No hay datos en la imagen"<<endl;
                return -2; //si no se pudo abrir, abortar ejecucion
            }
            //cout << image <<endl;
            //gen.addTrainImage(&image,p); //se agregan imagenes para el entrenamiento
            imagepaths.push_back(imagepath);
            imageclasses.push_back(p);
        }
    }
    for(int i = 0; i < imageclasses.size(); ++i) permindex.push_back(i); //dbg
    random_shuffle(permindex.begin(), permindex.end());
    for(int i : permindex) cout<<i<<" ";
    cout<<endl;

    vector<float> missed,tried; //etapa de clasificacion
    for(int iter=0 ; iter<10 ; iter++){ //10-fold
        BOFGenerator gen(nclasses); //nuestro generador de bag of words con tantas categorias como rutas pasadas como parametros

        for(int c=0; c<9; c++){ //etapa de diccionario
            for(int word_idx = c*chunksize ; word_idx < c*chunksize+chunksize ; word_idx++){
                cout << "DBG: " << imagepaths[permindex[word_idx]] << endl;
                image = imread( imagepaths[permindex[word_idx]], CV_LOAD_IMAGE_GRAYSCALE );
                gen.addWordImage(&image);
            }
        }
        gen.calculateWords(nwords);//finaliza etapa de diccionario

        for(int c=0; c<9; c++){ //etapa de entrenamiento
            for(int train_idx = c*chunksize ; train_idx < c*chunksize+chunksize ; train_idx++){
                cout << "DBG: " << imagepaths[permindex[train_idx]] << endl;
                image = imread( imagepaths[permindex[train_idx]], CV_LOAD_IMAGE_GRAYSCALE );
                gen.addTrainImage(&image,imageclasses[permindex[train_idx]]);
            }
        }

        if(algotype==1) gen.trainSIFT();//finaliza etapa de entrenamiento
        else gen.trainORB();

        int thismissed = 0, thistried = 0;
        for(int val_idx = 9*chunksize ; val_idx < 9*chunksize+chunksize ; val_idx++){
            thistried++;
            cout << "DBG: " << imagepaths[permindex[val_idx]] << endl;
            image = imread( imagepaths[permindex[val_idx]], CV_LOAD_IMAGE_GRAYSCALE );

            int classify;
            if(algotype==1) classify = gen.classifySIFT(&image,32);//finaliza etapa de entrenamiento
            else classify = gen.classifyBDC(&image,32);

            if( imageclasses[permindex[val_idx]] == classify ){ //se clasifican las imagenes
                //successful++;
                cout<<"DBG: clasificacion correcta!"<<endl;
            }else{
                thismissed++;
                cout<<"DBG: clasificacion erronea, se esperaba "<<imageclasses[permindex[val_idx]]<<endl;
            }
        }//finaliza etapa de clasificacion
        missed.push_back(thismissed);
        tried.push_back(thistried);
        cout<<"validate() iteracion "<<iter<<"terminada"<<endl;
        cout<<" intentados: "<<thistried<<" fracasos: "<<thismissed<<endl;
        cout<<" porcentaje de fracaso: "<<thismissed/thistried<<endl;
        rotate(permindex.begin(), permindex.begin() + chunksize, permindex.end());
    }
    float alltried = 0;
    cout<<"DBG: intentados por iter: "<<endl;
    for(float t : tried){
        cout<<t<<" ";
        alltried+=t;
    }
    cout<<endl;
    float allmissed = 0;
    cout<<"DBG: fracasos por iter: "<<endl;
    for(float m : missed){
        cout<<m<<" ";
        allmissed+=m;
    }
    cout<<endl;
    cout<<"DBG: intentos totales: "<<alltried<<" fracasos totales:"<<allmissed<<endl;

    return allmissed/alltried;
}

int main(int argc, char** argv )
{
    cout<<"Bag of words generator"<<endl<<"Alfonso R.O."<<endl;
    if ( (argc<3) || !(atoi(argv[argc-1])) || !(atoi(argv[argc-2])) )
    {
        cout << "parameters: /path/to/dataset1 [/path/to/dataset2] [...] number_of_images number_of_words train_algorithm " << endl;
        cout << "1 = SIFT. 2 = ORB." << endl;
        return -1; //validar parametros, imprimiendo ayuda en caso de no mas de 2 y enteros al final
    }
    float err= validate(argc,argv);
    cout<<"***PORCENTAJE DE ERROR:"<<err*100<<"%***"<<endl;
    return 0;
    /*
    Mat image; //declarar imagen de opencv
    int nimg = atoi(argv[argc-1]);
    string imagepath; //trabajar con strings es mas facil
    BOFGenerator gen(argc-2); //nuestro generador de bag of words con tantas categorias como rutas pasadas como parametros
    char numbers[4];

    for(int p=1; p<argc-1; p++){ //por cada dataset
        string imagedir(argv[p]); //cargar su path a una string
        for(int i=0; i<nimg; i++){ //cargar imagenes de la 1 a la number_of_images
            sprintf(numbers, "%04d", i+1); //convertir el indice en cadena digitos 0001-9999
            imagepath = imagedir + "/image_" + numbers + ".jpg"; //crear la direccion de la imagen
            cout << "DBG: " << imagepath << endl;
            image = imread( imagepath, CV_LOAD_IMAGE_GRAYSCALE ); //abrir imagen como escala de grises
            if ( !image.data )
            {
                cout<<"ERROR: No image data"<<endl;
                return -2; //si no se pudo abrir, abortar ejecucion
            }
            //cout << image <<endl;
            gen.addWordImage(&image); //calcular SIFTs de la imagen
        }
    }


    gen.calculateWords(200); //calcular un vocabulario de  500, cuanizando con k-means

    for(int p=1; p<argc-1; p++){ //por cada dataset
        string imagedir(argv[p]); //cargar su path a una string
        for(int i=0; i<nimg; i++){ //cargar imagenes de la 1 a la number_of_images
            sprintf(numbers, "%04d", i+1); //convertir el indice en cadena digitos 0001-9999
            imagepath = imagedir + "/image_" + numbers + ".jpg"; //crear la direccion de la imagen
            cout << "DBG: " << imagepath << endl;
            image = imread( imagepath, CV_LOAD_IMAGE_GRAYSCALE ); //abrir imagen como escala de grises
            if ( !image.data )
            {
                cout<<"ERROR: No image data"<<endl;
                return -2; //si no se pudo abrir, abortar ejecucion
            }
            //cout << image <<endl;
            gen.addTrainImage(&image,p); //se agregan imagenes para el entrenamiento
        }
    }

    gen.train(); //entrenar detector por k-nn

    float tried,successful;
    tried=0;
    successful=0;
    for(int p=1; p<argc-1; p++){ //por cada dataset
        cout<<endl;
        string imagedir(argv[p]); //cargar su path a una string
        for(int i=nimg+1; i<nimg+10; i++){ //cargar las 9 imagenes posteriores
            //for(int i=1; i<nimg+10; i++){ //DBG
            sprintf(numbers, "%04d", i); //convertir el indice en cadena digitos 0001-9999
            imagepath = imagedir + "/image_" + numbers + ".jpg"; //crear la direccion de la imagen
            cout << "DBG: " << imagepath << endl;
            image = imread( imagepath, CV_LOAD_IMAGE_GRAYSCALE ); //abrir imagen como escala de grises
            if ( !image.data )
            {
                cout<<"ERROR: No image data"<<endl;
                return -2; //si no se pudo abrir, abortar ejecucion
            }
            //cout << image <<endl;
            tried++;
            if( p == gen.classify(&image,32) ){ //se clasifican las imagenes
                successful++;
                cout<<"DBG: clasificacion correcta"<<endl;
            }else{
                cout<<"DBG: clasificacion erronea"<<endl;
            }
        }
    }

    cout<<"***PORCENTAJE DE PRECISION:"<<(successful/tried)*100<<"%***"<<endl;
    return 0;
*/
}



