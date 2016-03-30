/*
 * Generador de "bag of features"
 * objeto que se puede inicializar con parametros para el filtro SIFT o dejarse con la
 * configuracion por defecto.
 * Para usarse:
 * 1.-Inicializar objeto con alguno de los constructores (e.g. BOFGenerator obj(numclases) )
 * 2.-Agregar n imagenes en formato cv::Mat de opencv con obj.addWordImage(imagen).
 * 3.-Cuando se haya terminado, calcular el diccionario con obj.calculateWords(numpalabras).
 * 4.-Agregar imagenes para entrenar el clasificador con obj.addTrainImage(imagen,clase)
 * 5.-Entrenar el clasificador. Se usa trainSIFT() o bien trainORB()
 * 6.-Ahora, se puede clasificar con KNN usando obj.classify(imagen)
 *
 * Alfonso R.O.
 */

#ifndef BOFGENERATOR_H
#define BOFGENERATOR_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <fstream>

extern "C" {
#include <vl/sift.h>
#include <vl/kmeans.h>
}

using namespace std;

class BOFGenerator
{
private:
    ofstream myfile;//dbg
    vector<vl_sift_pix*> sifts; //los apuntadores a los descriptores SIFT
    vector<float*> matches; //matriz con los resultados; matches[imagen][ocurrenciadepalabra]
    vector<float*> probmatches; //matches con vectores probabilisticos
    vector<int> imgclass; //por cada imagen en matches, a que categoria corresponde?
    vector<int> descclust; //numero de cluster del par SIFT-ORB
    cv::Mat orbs; //descriptores ORB
    //vector<uint8_t>orbclass;
    cv::Mat probdesc; //palabras binarias probabilisticas
    VlKMeans *kmeans; //guardar objeto de k-means con los centros para cuantizar
    cv::ORB *orbDet; //detector ORB
    cv::SVM *svm; //clasificador support vector machine
    int nclasses; //numero de clases
    int nwords; //numero de palabras (numero de centros o clusters)
    int noct,nlvl,omin; //datos basicos para el SIFT
    bool cantrain; //bandera que define si es posible cuantizar
    bool cancategorize; //bandera que define si se puede hacer clasificacon

    void getDescriptors(VlSiftFilt *filt, vector<vl_sift_pix *> &sstorage, vector<cv::KeyPoint> &siftkps, int *lastkp); //metodo interno que calcula los SIFTs
    void processImage(cv::Mat * image , vector<vl_sift_pix *> &sstorage, vector<cv::KeyPoint> &siftKeypoints); //obtener sifts para image y guardarlos en sstorage
    //void trainBDC(); //entrenar clasificador binario
    void flattenSIFTMatrix(const vector<vl_sift_pix *> &in , vector<vl_sift_pix> &out); //aplanar matriz para uso en vlfeat

public:
    BOFGenerator(int nclass); //constructor por defecto: noctavas=-1, nniveles=3, ominima=0
    BOFGenerator(int nclass, int noctaves, int nlevels, int o_min, int nORB); //contructor con parametros: noctavas,nniveles,ominima
    //TODO:constructor con parametros threshold de orilla, pico y norm

    int addWordImage(cv::Mat * image); //agregar imagen para procesar
    int calculateWords(int ncenters); //generar vocabulario
    int addTrainImage(cv::Mat * image , int icla); //generar histograma por imagen para entrenamiento
    int trainSIFT(); //entrenar usando SIFT
    int trainORB(); //entrenar usando descriptores probabilisticos
    int classifySIFT(cv::Mat * image, int resolution); //clasificar
    int classifyBDC(cv::Mat * image , int resolution);

    ~BOFGenerator(); //destructor
};

#endif // BOFGENERATOR_H
