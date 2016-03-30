//usando informacion de http://synaptic-activity.blogspot.mx/2012/02/vlfeat-sift-with-opencv-code.html
//Alfonso R.O.



#include "bofgenerator.h"
#include <iomanip> //para imprimir los ORBs en hex

/***********************************************************************
 *
 *
 *-----------------------constructores y destructor---------------------
 *
 *
 *
 ***********************************************************************/

BOFGenerator::BOFGenerator(int nclass)
{
    cout<<"DBG: inicializando BOFGenerator con "<<nclass<<" clases"<<endl;
    orbDet = new cv::ORB(2000);
    noct = -1;
    nlvl = 3;
    omin = 0;
    nclasses=nclass;
    cantrain = false;
    cancategorize = false;
    myfile.open ("histogramas.csv");//dbg
    //clahist.reserve(nclasses);
}

BOFGenerator::BOFGenerator(int nclass, int noctaves, int nlevels, int o_min, int nORB)
{
    orbDet = new cv::ORB(nORB);
    noct = noctaves;
    nlvl = nlevels;
    omin = o_min;
    nclasses=nclass;
    cantrain = false;
    cancategorize = false;
    //clahist.reserve(nclasses);
}

BOFGenerator::~BOFGenerator() //destructor
{

    vl_kmeans_delete(kmeans);

    /*
    size_t killsize=clahist.size();
    for(int j = 0; j <killsize; j++ ) {
        //cout<<*sifts[i]<<endl;
        delete[] clahist[j] ; //eliminar histogramas porque son dinamicamente creados
    }
    */

    //killsize=sifts.size();
    for(vl_sift_pix *i : sifts){
        //for(int i = 0; i <killsize; i++ ) {
        delete[] i;
        //    delete[] sifts[i] ; //eliminar sifts porque son dinamicamente creados
    }

    //killsize=matches.size();
    //for(int k = 0; k <killsize; k++ ) {
    for(float *i : matches){
        //cout<<*sifts[i]<<endl;
        delete[] i ; //eliminar sifts porque son dinamicamente creados
    }

    delete svm;
    delete orbDet;
    myfile.close();//dbg

    cout<<"DBG: memoria liberada"<<endl;
}

/***********************************************************************
 *
 *
 *---------------------------metodos publicos---------------------------
 *
 *
 *
 ***********************************************************************/

int BOFGenerator::addWordImage(cv::Mat *image)
{
    if(cantrain)return -1; //no agregar mas imagenes


    vector<vl_sift_pix*> thisSift;
    vector<cv::KeyPoint> skps;//kps sift


    cv::Mat thisOrb; //descriptores ORB para esta imagen
    processImage(image,thisSift,skps); //obtener sifts de la imagen.

    cout << "BOFGenerator::addWordImage(): "<< skps.size() << " SIFT kps antes de ORB" << endl;

    cv::Mat mask(image->size(), CV_8UC1, cv::Scalar::all(255)); //el detector necesita una mask
    (*orbDet)(*image,mask,skps,thisOrb,true); //Obtener descriptores orb

    cout << "BOFGenerator::addWordImage(): "<< skps.size() << " SIFT kps despues de ORB" << endl;

    cout << "BOFGenerator::addWordImage(): extraidos "<< thisSift.size() << " SIFTs y " << thisOrb.rows << " ORBs" << endl;

    int currsift = 0 , currkp = 0;

    for (float * sift : thisSift){

        if( (currkp==skps.size()) || (skps[currkp].class_id!=currsift) ){ //si ya no hay mas keypoints o el sift no corresponde al indice del keypoint
            delete[] sift; //destruir el descriptor actual
            ++currsift;
            //cout << currsift << " muere" <<endl;
            continue;
        }

        //si lo de arriba no se cumple...
        sifts.push_back(sift); //se agrega el descriptor a la tabla
        //cout << currsift << " vive" <<endl;
        ++currkp;
        ++currsift;

    }

    /*
    for(cv::KeyPoint kp : skps){//destruir los sifts innecesarios, agregar sobrevivientes a la tabla

        //cout<<kp.class_id<<" ";
        if(kp.class_id != arrpoint){
            while(arrpoint != kp.class_id){
                delete[] thisSift[arrpoint];
                ++arrpoint;
                cout << arrpoint << " muere" <<endl;
            }
        }else{
            sifts.push_back(thisSift[kp.class_id]);
            ++arrpoint;
            cout << kp.class_id << " vive" <<endl;
        }

    }
    */

    //cout <<endl;
    orbs.push_back(thisOrb); //agregar orbs a la tabla de la clase

    //orbclass.resize(orbclass.size()+thisOrb.rows , icla); //todos los orbs agregados pertenecen a una clase


    cout << "BOFGenerator::addWordImage(): "<< sifts.size() << " SIFTs y " << orbs.rows << " ORBs en las tablas" << endl;
    //sifts.insert(sifts.end() , thisSift.begin() , thisSift.end() );


    return 0;


}

int BOFGenerator::calculateWords(int ncenters)
{
    if(cantrain)return -1; //ya se generaron palabras

    cout<<"BOFGenerator::calculateWords(): inicializando kmeans..."<<endl;
    kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2); //inicializar objeto k-means
    vl_kmeans_set_algorithm(kmeans,VlKMeansElkan); //usar algoritmo de Elkan. Segun la documentacion, es el preferido. ANN es mas rapido pero menos exacto
    //vl_kmeans_set_algorithm(kmeans,VlKMeansLloyd);
    vl_kmeans_set_initialization(kmeans,VlKMeansPlusPlus); //usar inicializacion diversificada
    //vl_kmeans_set_initialization(kmeans,VlKMeansRandomSelection);
    //vl_kmeans_set_num_repetitions(kmeans,3); //numero de repeticiones
    //vl_kmeans_set_max_num_iterations(kmeans,100); //numero de iteraciones por repeticion
    //vl_kmeans_set_num_trees(kmeans,3);//arboles de busqueda?
    vl_kmeans_set_verbosity(kmeans,1); //debug

    //kmeans necesita un arreglo unidimensional
    vector<vl_sift_pix> asifts;
    int ds = sifts.size();
    flattenSIFTMatrix(sifts,asifts);

    //ejecutar k-means
    cout<<"BOFGenerator::calculateWords(): ejecutando k-means para "<<ds<<" descriptores..."<<endl;
    double energ=vl_kmeans_cluster(kmeans,&asifts[0],128,ds,ncenters);
    cout<<"BOFGenerator::calculateWords(): se han obtenido " << kmeans->numCenters << " clusters" <<endl;
    cout<<"BOFGenerator::calculateWords(): energia de la solucion: " << energ <<endl; //no se para que sirva

    //debug
    //        const vl_sift_pix *cnt = (vl_sift_pix*)vl_kmeans_get_centers(kmeans);
    //        for (int i = 0; i < ncenters; i++) {
    //            cout<<"Centro #"<< i<<endl;
    //            for (int j = 0; j < 128; j++) {
    //                cout<<"    coord["<<j<<"] = "<< cnt[128 * i + j]<<endl;
    //            }
    //            getchar();
    //        }
    nwords=ncenters;
    cout<<"calculateWords(): cuantizando sifts..."<<endl;
    vl_uint32 * assi = new vl_uint32[ds]();
    vl_kmeans_quantize(kmeans,assi,NULL,&asifts[0],ds); //se hace la cuantizacion

    cout<<"calculateWords(): asignando etiquetas..."<<endl;
    for(int as=0;as<ds;as++){//por cada asignacion

        descclust.push_back(assi[as]); //etiquetar el par de descriptores
        cout<<assi[as]<<" ";
    }
    cout<<endl;

    //inicia calculo de vectores probabilisticos
    probdesc = cv::Mat::zeros(nwords,256,CV_32FC1);
    vector<cv::Mat>matrixperword (nwords); //grupo de ORBs que estan emparentados a un cluster SIFT
    fill(matrixperword.begin() , matrixperword.end() , cv::Mat()); //llenar vector con headers de matrices vacias
    int currpair=0; //par actual
    for (int clust : descclust){

        //TODO: cual funka mejor
        //orbs.row(currpair++).copyTo(matrixperword[clust]); //generar matrixperword
        matrixperword[clust].push_back(orbs.row(currpair++));
    }

    int clust = 0;
    const uint8_t bitmask = 1; //mask para la operacion de bitwise and

    for (cv::Mat wordmatrix : matrixperword){ //por cada cluster de palabras

        for(int descbyte = 0 ; descbyte<orbs.cols ; descbyte++){ //por cada uno de los 32 bytes

            for(int descbit = 0 ; descbit<8 ; descbit++){//por cada bit

                float nonzero; //cuantos elementos no son cero
                cv::Mat countvector; //Donde see guarda el resultado del and
                cv::bitwise_and( wordmatrix.col(descbyte) , cv::Mat( wordmatrix.rows , 1 , CV_8UC1 , cv::Scalar(bitmask<<descbit) ) , countvector ); //hacer un and entre todos los bits del descriptor 32x8 para que se pueda contar
                //TODO: los de abajo serviran?
                //cv::bitwise_and( wordmatrix.col(descbyte) , cv::Scalar::all(bitmask<<descbit) , countvector ); //hacer un and entre todos los bits del descriptor 32x8 para que se pueda contar
                nonzero = cv::countNonZero(countvector) / float(countvector.rows); //contar unos y normalizar
                probdesc.at<float>(clust , descbit+(descbyte*8) ) = nonzero;

                //dbg
                //cout << nonzero << " " ;

            }
        }
        //cout<<endl; //dbg
        clust++;
    }

    //dbg
    if(orbs.rows != sifts.size()){
        cout<<"BOFGenerator::calculateWords(): ERROR "
           << "orbs:" << orbs.rows << "," << orbs.cols << " "
           << "sifts:"<< sifts.size() << " "
           << "labels:" << descclust.size() << endl;
        getchar();
    }

    cantrain=true;//activar bandera para permitir cuantizacion
    //delete[] asifts; //liberar memoria
    delete[] assi;
    return 0;
}

int BOFGenerator::addTrainImage(cv::Mat *image, int icla)
{
    if(!cantrain) return -1; //aun no hay vocabulario
    cout<<"BOFGenerator::addTrainImage(): clase: "<<icla<<" nclases: "<<nclasses<<endl;
    if( icla<1 || icla>nclasses) return -2; //existe esa clase?
    if(cancategorize) return -3; //ya esta normalizado

    cout<<"BOFGenerator::addTrainImage(): obteniendo descriptores de la imagen a cuantizar..."<<endl;
    vector<vl_sift_pix*> thisSift; //vector para guardar los SIFT
    //vector<cv::KeyPoint> dummy; //matriz dummy para guardar kps
    vector<cv::KeyPoint> skps;
    processImage(image,thisSift,skps);

    ///
    ///
    /// Obtener histogramas probabilisticos
    ///
    ///

    cout<<"BOFGenerator::addTrainImage():: generando histograma ORB..."<<endl;

    cv::Mat thisOrb; //descriptores ORB para esta imagen
    cv::Mat mask(image->size(), CV_8UC1, cv::Scalar(255)); //el detector necesita una mask
    (*orbDet)(*image,mask,skps,thisOrb,true); //Obtener descriptores orb


    cv::Mat cflatOrb(thisOrb.rows , 1 ,CV_8UC1 ,cv::Scalar(0)); //columna dummy para concatenación. opencv no permite concatenacion horizontal con matriz vacía
    cv::Mat flatOrb; //se expande el orb


    const uint8_t bitmask = 1; //mask para la operacion de bitwise and

    for(int descbyte = 0 ; descbyte<thisOrb.cols ; descbyte++){ //por cada uno de los 32 bytes

        for(uint8_t descbit = 0 ; descbit<8 ; descbit++){//por cada bit

            cv::Mat bitcol; //para guardar cada columna separada por bits
            //cv::bitwise_and( thisOrb.col(descbyte) , cv::Scalar::all(bitmask<<descbit) , bitcol ); //hacer un and entre todos los bits del descriptor 32x8 para aplanarlo. un bit por celda
            cv::bitwise_and( thisOrb.col(descbyte) , cv::Mat( thisOrb.rows , 1 , CV_8UC1 , cv::Scalar(bitmask<<descbit) ) , bitcol );
            cv::hconcat(cflatOrb , bitcol , cflatOrb);
        }
    }

    cflatOrb.colRange(1,257).convertTo(flatOrb , CV_32FC1); //recortar la columna dummy y convertir a float

    cv::threshold(flatOrb , flatOrb ,1.0 , 1.0 , CV_THRESH_BINARY); //convertir valores no nulos a 1

    //DBG
    //    for(int i = 0; i<flatOrb.rows ;i++){
    //        for(int j = 0; j<flatOrb.cols ;j++){
    //            cout << flatOrb.at<float>(i,j) << " ";
    //        }
    //        cout << endl;
    //    }

    float *probimghist = new float[nwords](); //histograma probabilistico

    for(int desc = 0 ; desc<flatOrb.rows ; desc++){

        double maxprob = 0; //la mayor probabilidad
        int maxclust = 0; //cluster con la mayor probabilidad

        for(int clust = 0 ; clust<probdesc.rows ; clust++){

            cv::Mat sumvec1 = probdesc.row(clust).mul(flatOrb.row(desc)); //multiplicacion elementwise (probabilidad de unos)

            cv::Mat cprobdesc; //complemento del descriptor probabilistico
            cv::Mat cflatOrb; //complemento del descriptor ORB
            //cv::absdiff(flatOrb.row(desc) , cv::Scalar::all(-1.0) , cflatOrb);
            cv::absdiff(flatOrb.row(desc) , cv::Mat( 1 , 256 , CV_32FC1 , cv::Scalar(1.0) ) , cflatOrb);
            //cv::absdiff(probdesc.row(clust) , cv::Scalar::all(-1.0) , cprobdesc);
            cv::absdiff(probdesc.row(clust) , cv::Mat( 1 , 256 , CV_32FC1 , cv::Scalar(1.0) ) , cprobdesc);
            cv::Mat sumvec2 = cprobdesc.mul(cflatOrb); //multiplicacion elementwise con el complemento (probabilidad de ceros)

            double totprob = cv::sum(sumvec1 + sumvec2)[0]; //probabilidades de ceros y unos se suman para obtener la probabilidad total
            if(totprob > maxprob){//si la probabilidad de que este descriptor pertenezca a este cluster es mayor que la ultima mas grande

                maxclust = clust; //este cluster es de de mayor probabilidad
                maxprob = totprob; //esta es la mayor probabilidad
            }
        }

        //al final del ciclo, se tiene a que cluster pertence
        probimghist[maxclust]++;
        //cout << maxclust <<" ";

    }

    //cout <<endl;


    for(int i=0;i<nwords;i++){ //dbg
        //myfile << imghist[i] <<",";
        cout << probimghist[i] <<" ";
    }
    cout<<endl;

    probmatches.push_back(probimghist); //un histograma de palabras probabilisticas

    ///
    ///
    /// Obtener histogramas SIFT
    ///
    ///


    const size_t ds = thisSift.size(); //numero de SIFTs
    vl_uint32 * assi = new vl_uint32[ds]();  //numero de cluster asignado
    //float * dist = new float[ds]; //la distancia de cada SIFT al centro del cluster asignado. No se usan

    cout<<"BOFGenerator::addTrainImage():: preparando "<<ds<<" descriptores para cuantizacion..."<<endl;
    //kmeans necesita un arreglo unidimensional transpuesto
    vector<vl_sift_pix> asifts;
    flattenSIFTMatrix(thisSift,asifts);

    cout<<"BOFGenerator::addTrainImage():: cuantizando imagen..."<<endl;
    vl_kmeans_quantize(kmeans,assi,NULL,&asifts[0],ds); //se hace la cuantizacion

    float *imghist = new float[nwords]();
    cout<<"BOFGenerator::addTrainImage():: generando histograma SIFT..."<<endl;

    for(int as=0;as<ds;as++){//por cada asignacion

        imghist[assi[as]]++; //sumar una ocurrencia de palabra
        if(assi[as]>(nwords-1)){ //dbg

            cout<<"ERROR EN BOFGenerator::addTrainImage():"<<assi[as]<<endl;
            getchar();
        }
    }


    for(int i=0;i<nwords;i++){ //dbg
        //myfile << imghist[i] <<",";
        cout << imghist[i] <<" ";
    }
    //myfile << endl;
    cout<<endl;

    matches.push_back(imghist); //un histograma de palabras para una imagen...
    imgclass.push_back(icla); //de cierta clase

    //delete[] asifts; //liberar memoria
    delete[] assi;
    //delete[] dist;
    cout<<"BOFGenerator::addTrainImage(): terminada cuantizacion de la imagen"<<endl;
    return 0;
}

int BOFGenerator::trainSIFT()
{
    if(!cantrain) return -1; //aun no hay diccionario
    if(cancategorize) return -2; //ya se hizo el entrenamiento
    cancategorize=true; //ya se puede categorizar

    cout<<"-----ENTRENAMIENTO-----"<<endl;
    //KNearest de opencv necesita trabajar con Mat's
    cv::Mat traindata(matches.size(),nwords,CV_32FC1); //int 32bits signed 1 channel

    for(int i=0;i<traindata.rows;i++){

        for (int j=0;j<traindata.cols;j++){

            traindata.at<float>(i,j) = (matches[i])[j];
            cout<<traindata.at<float>(i,j)<<" ";//dbg
        }
        cout<<endl; //dbg
    }
    cout<<"BOFGenerator::trainSIFT(): traindata. rows:"<<traindata.rows<<" cols:"<<traindata.cols<<endl;//dbg

    cout<<"-----RESPUESTAS-----"<<endl;
    //int *imp=imgclass.data();
    cv::Mat responses(imgclass.size(),1,CV_32SC1);

    for(int i=0;i<responses.rows;i++){

        responses.at<int>(i,0) = (int)imgclass[i];
        cout<<responses.at<int>(i,0)<<endl;//dbg
    }

    cout<<endl; //dbg
    cout<<"BOFGenerator::trainSIFT(): responses. rows:"<<responses.rows<<" cols:"<<responses.cols<<endl;//dbg

    cout<<"BOFGenerator::trainSIFT(): entrenando SVM"<<endl;
    //knn=new cv::KNearest; //inicializar objeto de k vecinos cercanos
    //knn->train(traindata,responses);
    cv::SVMParams params; //parametros de entrenamiento del svm. obtenidos de http://docs.opencv.org/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
    //params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::LINEAR;

    svm = new cv::SVM();
    svm->train_auto(traindata,responses,cv::Mat(),cv::Mat(),params);
    cout<<"BOFGenerator::trainSIFT(): entrenamiento terminado"<<endl;
    //cout<<"DBG: k_max= "<<knn->get_max_k()<<endl<<" samples:"<<knn->get_sample_count()<<" vars:"<<knn->get_var_count()<<endl;
    //getchar();//dbg

    return 0;
}

int BOFGenerator::trainORB()
{
    if(!cantrain) return -1; //aun no hay diccionario
    if(cancategorize) return -2; //ya se hizo el entrenamiento
    cancategorize=true; //ya se puede categorizar

    cout<<"-----ENTRENAMIENTO-----"<<endl;
    //KNearest de opencv necesita trabajar con Mat's
    cv::Mat traindata(probmatches.size(),nwords,CV_32FC1); //int 32bits signed 1 channel

    for(int i=0;i<traindata.rows;i++){

        for (int j=0;j<traindata.cols;j++){

            traindata.at<float>(i,j) = (probmatches[i])[j];
            cout<<traindata.at<float>(i,j)<<" ";//dbg
        }
        cout<<endl; //dbg
    }
    cout<<"BOFGenerator::trainORB(): traindata. rows:"<<traindata.rows<<" cols:"<<traindata.cols<<endl;//dbg

    cout<<"-----RESPUESTAS-----"<<endl;
    //int *imp=imgclass.data();
    cv::Mat responses(imgclass.size(),1,CV_32SC1);

    for(int i=0;i<responses.rows;i++){

        responses.at<int>(i,0) = (int)imgclass[i];
        cout<<responses.at<int>(i,0)<<endl;//dbg
    }

    cout<<endl; //dbg
    cout<<"BOFGenerator::trainORB(): responses. rows:"<<responses.rows<<" cols:"<<responses.cols<<endl;//dbg

    cout<<"BOFGenerator::trainORB(): entrenando SVM"<<endl;
    //knn=new cv::KNearest; //inicializar objeto de k vecinos cercanos
    //knn->train(traindata,responses);
    cv::SVMParams params; //parametros de entrenamiento del svm. obtenidos de http://docs.opencv.org/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
    //params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::LINEAR;

    svm = new cv::SVM();
    svm->train_auto(traindata,responses,cv::Mat(),cv::Mat(),params);
    cout<<"BOFGenerator::trainORB(): entrenamiento terminado"<<endl;
    //cout<<"DBG: k_max= "<<knn->get_max_k()<<endl<<" samples:"<<knn->get_sample_count()<<" vars:"<<knn->get_var_count()<<endl;
    //getchar();//dbg

    return 0;
}

int BOFGenerator::classifySIFT(cv::Mat *image, int resolution)
{
    if(!cancategorize) return -1;

    cout<<"BOFGenerator::classifySIFT() : obteniendo descriptores de la imagen a clasificar..."<<endl;
    vector<vl_sift_pix*> imgsifts; //vector para guardar los SIFT
    vector<cv::KeyPoint> dummy; //matriz dummy para guardar orbs
    processImage(image,imgsifts,dummy); //obtener sifts de la imagen
    //TODO: porque no se puede esto?
    //processImage(image,imgsifts,cv::Mat());

    size_t ds = imgsifts.size(); //numero de SIFTs
    vl_uint32 * assi = new vl_uint32[ds]();  //numero de cluster asignado
    //float * dist = new float[ds]; //la distancia de cada SIFT al centro del cluster asignado. No se usan

    cout<<"BOFGenerator::classifySIFT() : preparando "<<ds<<" descriptores para cuantizacion..."<<endl;
    //kmeans necesita un arreglo unidimensional transpuesto
    vector<vl_sift_pix> asifts;
    flattenSIFTMatrix(imgsifts,asifts);


    cout<<"BOFGenerator::classifySIFT() : cuantizando imagen..."<<endl;
    vl_kmeans_quantize(kmeans,assi,NULL,&asifts[0],ds); //se hace la cuantizacion

    float *imghist = new float[nwords]();
    cout<<"BOFGenerator::classifySIFT() : generando histograma..."<<endl;

    for(int as=0;as<ds;as++){//por cada asignacion

        imghist[assi[as]]++; //sumar una ocurrencia de palabra
        if(assi[as]>(nwords-1) || assi[as]<0){ //dbg

            cout<<"ERROR EN BOFGenerator::classify() :"<<assi[as]<<endl;
            getchar();
        }
    }

    cv::Mat catdata(1,nwords,CV_32FC1);
    for(int j=0;j<catdata.cols;j++){

        catdata.at<float>(0,j) = imghist[j];
        cout<<catdata.at<float>(0,j)<<" ";//dbg
        //myfile << catdata.at<float>(0,j) <<",";
    }
    //myfile << endl;
    cout<<endl;
    cout<<"BOFGenerator::classifySIFT() : catdata. rows:"<<catdata.rows<<" cols:"<<catdata.cols<<endl;//dbg
    //cv::Mat results;//la clase!!!
    //knn->find_nearest(catdata,20,results);

    //cv::Mat results,responses;
    //nearest = cv::Mat::zeros(1, resolution, CV_32FC1);
    //int theclass=knn->find_nearest(catdata,resolution,results,responses);
    int theclass = svm->predict(catdata,true);

    cout<<"***LA CLASE DETECTADA ES "<<theclass<<"***"<<endl<<endl<<endl;
    /*
    cout<<"results:"<<endl;
    for(int j=0;j<results.cols;j++){
        cout<<results.at<float>(0,j)<<" ";//dbg
    }
    cout<<endl;
    cout<<"responses:"<<endl;
    for(int j=0;j<responses.cols;j++){
        cout<<responses.at<float>(0,j)<<" ";//dbg
        //cout<<responses.data[j]<<" ";//dbg
    }
    cout<<endl;

    cout<<"DBG: results. rows:"<<results.rows<<" cols:"<<results.cols<<endl;//dbg
    cout<<"DBG: responses. rows:"<<responses.rows<<" cols:"<<responses.cols<<endl;//dbg
    //cout<<"DBG: distances. rows:"<<distances.rows<<" cols:"<<distances.cols<<endl;//dbg
    */

    //delete[] asifts; //liberar memoria
    delete[] assi;
    delete[] imghist;
    //delete[] dist;
    return theclass;
}

int BOFGenerator::classifyBDC(cv::Mat * image , int resolution)
{
    if(!cancategorize) return -1;

    cout<<"BOFGenerator::classifyBCD() : obteniendo descriptores de la imagen a clasificar..."<<endl;


    cv::Mat thisOrb; //descriptores ORB para esta imagen
    vector<cv::KeyPoint> orbkps; //dummy para los kps
    cv::Mat mask(image->size(), CV_8UC1, cv::Scalar(255)); //el detector necesita una mask
    (*orbDet)(*image,mask,orbkps,thisOrb,false); //Obtener descriptores orb


    cv::Mat cflatOrb(thisOrb.rows , 1 ,CV_8UC1 ,cv::Scalar(0)); //columna dummy para concatenación. opencv no permite concatenacion horizontal con matriz vacía
    cv::Mat flatOrb; //se expande el orb


    cout<<"BOFGenerator::classifyBCD() : preparando "<<thisOrb.rows<<" descriptores para cuantizacion..."<<endl;

    const uint8_t bitmask = 1; //mask para la operacion de bitwise and

    for(int descbyte = 0 ; descbyte<thisOrb.cols ; descbyte++){ //por cada uno de los 32 bytes

        for(uint8_t descbit = 0 ; descbit<8 ; descbit++){//por cada bit

            cv::Mat bitcol; //para guardar cada columna separada por bits
            //cv::bitwise_and( thisOrb.col(descbyte) , cv::Scalar::all(bitmask<<descbit) , bitcol ); //hacer un and entre todos los bits del descriptor 32x8 para aplanarlo. un bit por celda
            cv::bitwise_and( thisOrb.col(descbyte) , cv::Mat( thisOrb.rows , 1 , CV_8UC1 , cv::Scalar(bitmask<<descbit) ) , bitcol );
            cv::hconcat(cflatOrb , bitcol , cflatOrb);
        }
    }

    cflatOrb.colRange(1,257).convertTo(flatOrb , CV_32FC1); //recortar la columna dummy y convertir a float

    cv::threshold(flatOrb , flatOrb ,1.0 , 1.0 , CV_THRESH_BINARY); //convertir valores no nulos a 1

    //DBG
    //    for(int i = 0; i<flatOrb.rows ;i++){
    //        for(int j = 0; j<flatOrb.cols ;j++){
    //            cout << flatOrb.at<float>(i,j) << " ";
    //        }
    //        cout << endl;
    //    }

    float *probimghist = new float[nwords](); //histograma probabilistico

    for(int desc = 0 ; desc<flatOrb.rows ; desc++){

        double maxprob = 0; //la mayor probabilidad
        int maxclust = 0; //cluster con la mayor probabilidad

        for(int clust = 0 ; clust<probdesc.rows ; clust++){

            cv::Mat sumvec1 = probdesc.row(clust).mul(flatOrb.row(desc)); //multiplicacion elementwise (probabilidad de unos)

            cv::Mat cprobdesc; //complemento del descriptor probabilistico
            cv::Mat cflatOrb; //complemento del descriptor ORB
            //cv::absdiff(flatOrb.row(desc) , cv::Scalar::all(-1.0) , cflatOrb);
            cv::absdiff(flatOrb.row(desc) , cv::Mat( 1 , 256 , CV_32FC1 , cv::Scalar(1.0) ) , cflatOrb);
            //cv::absdiff(probdesc.row(clust) , cv::Scalar::all(-1.0) , cprobdesc);
            cv::absdiff(probdesc.row(clust) , cv::Mat( 1 , 256 , CV_32FC1 , cv::Scalar(1.0) ) , cprobdesc);
            cv::Mat sumvec2 = cprobdesc.mul(cflatOrb); //multiplicacion elementwise con el complemento (probabilidad de ceros)

            double totprob = cv::sum(sumvec1 + sumvec2)[0]; //probabilidades de ceros y unos se suman para obtener la probabilidad total
            if(totprob > maxprob){//si la probabilidad de que este descriptor pertenezca a este cluster es mayor que la ultima mas grande

                maxclust = clust; //este cluster es de de mayor probabilidad
                maxprob = totprob; //esta es la mayor probabilidad
            }
        }

        //al final del ciclo, se tiene a que cluster pertence
        probimghist[maxclust]++;
        //cout << maxclust <<" ";

    }



    cv::Mat catdata(1,nwords,CV_32FC1);
    for(int j=0;j<catdata.cols;j++){

        catdata.at<float>(0,j) = probimghist[j];
        cout<<catdata.at<float>(0,j)<<" ";//dbg
        //myfile << catdata.at<float>(0,j) <<",";
    }
    //myfile << endl;
    cout<<endl;
    cout<<"BOFGenerator::classifyBCD() : catdata. rows:"<<catdata.rows<<" cols:"<<catdata.cols<<endl;//dbg
    //cv::Mat results;//la clase!!!
    //knn->find_nearest(catdata,20,results);

    //cv::Mat results,responses;
    //nearest = cv::Mat::zeros(1, resolution, CV_32FC1);
    //int theclass=knn->find_nearest(catdata,resolution,results,responses);
    int theclass = svm->predict(catdata,true);

    cout<<"***LA CLASE DETECTADA ES "<<theclass<<"***"<<endl<<endl<<endl;

    delete[] probimghist;
    //delete[] dist;
    return theclass;
}


/***********************************************************************
 *
 *
 * ---------------------------metodos privados--------------------------
 *
 *
 *
 ***********************************************************************/

void BOFGenerator::getDescriptors(VlSiftFilt *filt, vector<vl_sift_pix *> &sstorage, vector<cv::KeyPoint> &siftkps, int* lastkp)
{
    size_t sl=128;//size del sift
    double angles[4]; //buffer para guardar las orientaciones actuales
    VlSiftKeypoint const *keys = vl_sift_get_keypoints(filt); //todos los keypoints
    VlSiftKeypoint const *kp; //apuntador a los keypoints en la octava actual
    int nkp = vl_sift_get_nkeypoints(filt); //numero de keypoints

    for(int k=0 ; k<nkp ; k++){ //por cada keypoint en esta octava

        kp = keys+k;

        int norient = vl_sift_calc_keypoint_orientations(filt,angles,kp); //calcular las orientaciones de los keypoints
        //cout << "DBG: encontradas " << norient << " orientaciones para el keypoint "<< k <<endl;
        for(int i =0;i<norient;i++){
            //tomar solo el primer angulo
            vl_sift_pix *newsift = new vl_sift_pix [sl]();//crear descriptor vacio

            vl_sift_calc_keypoint_descriptor(filt,newsift,kp,angles[i]); //calcular el descriptor SIFT

            //TODO: esta bien el size del kp?

            //extraer info del keypoint para el descriptor orb
            //cv::KeyPoint *orbkp;
            //siftkps.push_back( KeyPoint(kp->ix, kp->iy, 2*kp->sigma, angles[i]*(180/M_PI), 0, kp->o) );
            siftkps.push_back( cv::KeyPoint(kp->x, kp->y, 2*kp->sigma, angles[i]*(180/M_PI), 0, kp->o ,(*lastkp)++ ) );

            bool isNull = false;
            int nnull=0;
            vl_sift_pix feat;//PRUEBA
            for(int j=0;j<128;j++){ //TODO: ver si los descriptores vacios son un bug

                nnull += ( newsift[j]==0 ? 1 : 0);

                if( nnull>126 ){

                    //cout<<"ALERTA EN newsift[]: "<<nnull<<" ceros"<<endl;
                    isNull = true;
                    //getchar();
                }
                //if( ( newsift[j]>1) || ( newsift[j]<0) ){
                if( ( newsift[j]>1) || ( newsift[j]<0) ){
                    cout<<"ALERTA EN newsift[]: "<<newsift[j]<<endl;
                    getchar();
                }
                feat=newsift[j]*512.0F;//PRUEBA
                newsift[j] = (feat<255.0F) ? floor(feat) : 255.0F;//PRUEBA

                //cout << "DBG: descriptor SIFT,elemento " << j << ": " << newsift[j] <<endl;
            }

            if(isNull){
                cout<<"BOFGenerator::getDescriptors(): descriptor nulo"<<endl;

            }

            sstorage.push_back( newsift );
        }
    }
}

//ORIGINAL
/*
void BOFGenerator::processImage(cv::Mat * image , vector<vl_sift_pix *> &sstorage, cv::Mat &ostorage)
{
    int w=image->size().width;
    int h=image->size().height;
    int lastkp = 0; //id del keypoint para poderlos numerar
    //size_t imb = image->cols * image->rows;
    //vl_sift_pix *imdata = new vl_sift_pix[imb];
    vector<vl_sift_pix> imdata;
    vector<cv::KeyPoint> orbKeypoints;
    //cout << "processImage(): convirtiendo imagen de "<<w<<"x"<< h<<". npixels: "<< imb <<" nchannels:"<<image->channels()<<" step:"<<image->step<<endl;

    //ROW MAJOR
    for (int row = 0; row < image->rows; row++){
        for (int col = 0; col < image->cols; col++){
            imdata.push_back( (vl_sift_pix)(image->at<uchar>(row,col))/255.0 );
            //imdata.push_back( ((vl_sift_pix)ldata[j*image->step+i]) );
            //cout<<imdata[row*image->cols+col]<<" ";
        }
        //cout<<endl;
        //getchar();
    }

    //COL MAJOR

//    for (int col = 0; col < image->cols; col++){
//        for (int row = 0; row < image->rows; row++){
//            imdata.push_back( (vl_sift_pix)(image->at<uchar>(row,col))/255.0 );
//            //imdata.push_back( ((vl_sift_pix)ldata[j*image->step+i]) );
//            //cout<<imdata[row*image->cols+col]<<" ";
//        }
//        //cout<<endl;
//        //getchar();
//    }

    //toda esta porqueria es para que vlfeat pueda trabajar con imagenes de opencv
    //vlfeat tiene una convencion anormal de notacion matricial. Ver "column mayor"
    //ademas, la matriz tiene que aplanarse a un array unidimensional


    cout << "processImage(): inicializando filtro..." << endl;
    VlSiftFilt *filt = vl_sift_new(w , h , noct , nlvl , omin);  //inicializar filtro SIFT


    //cout << "DBG: procesando primera octava (0)..." << endl;
    //int retval = vl_sift_process_first_octave(filt , imdata); //procesar 1era octava
    int retval = vl_sift_process_first_octave(filt , &imdata[0]);
    cout << "processImage(): encontradas "<< (filt->O) <<" octavas"<<endl;

    while(1){ //mientras haya octavas que procesar

        if(VL_ERR_EOF == retval){ //procesar siguente octava
            break; //si no hay mas, terminar
        }
        //cout << "DBG: procesando octava "<< vl_sift_get_octave_index(filt) << "..." << endl;

        vl_sift_detect(filt); //detectar keypoints
        //cout << "DBG: encontrados " << (vl_sift_get_nkeypoints(filt)) <<" keypoints"<< endl;

        getDescriptors(filt,sstorage,orbKeypoints,&lastkp); //calcular descriptores
        retval = vl_sift_process_next_octave(filt);
        //retval=VL_ERR_EOF; //forzar una sola octava
    }
    //cout<<"processImage(): calculados todos los descriptores SIFT de esta imagen"<<endl<<endl;
    //cout<<"processImage(): hay "<< orbKeypoints.size() << " ORB kps" <<endl;

    //cout <<"processImage(): calculando descriptores ORB"<<endl;
    orb(*image,cv::noArray(),orbKeypoints,ostorage,true); //calcular los descriptores para los keypoints obtenidos por el detector de vlfeat
    //orb.compute(*image,orbKeypoints,ostorage);
    //dbg
    //if(ostorage.rows != orbKeypoints.size()){
    cout << "processImage(): "
         << "ostorage:" << ostorage.rows << ","<<ostorage.cols<<", "
         << "orbKeypoints" << orbKeypoints.size()<<endl;
    //getchar();
    //}

    cout.setf(ios_base::uppercase); //para imprimir hex
    for(int i=0;i<ostorage.rows;i++){
        cout<<"0x";
        for (int j=0;j<ostorage.cols;j++){
            cout<<hex<<setw(2)<<setfill('0')<<(uint32_t)ostorage.at<uint8_t>(i,j);//dbg
        }
        cout<<dec<<endl; //dbg

        //cout<<"DBG: agregado descriptor a la lista"<<endl;
    }

    vl_sift_delete(filt); //destruir filtro para liberar memoria
    //delete[] imdata;
}
*/
//NUEVA

void BOFGenerator::processImage(cv::Mat * image , vector<vl_sift_pix *> &sstorage, vector<cv::KeyPoint> &siftKeypoints)
{
    int w=image->size().width;
    int h=image->size().height;
    int lastkp = 0; //id del keypoint para poderlos numerar
    //size_t imb =  w*h ;
    size_t imb = image->cols * image->rows;
    //vl_sift_pix *imdata = new vl_sift_pix[imb];
    vector<vl_sift_pix> imdata;

    //cout << "processImage(): convirtiendo imagen de "<<w<<"x"<< h<<". npixels: "<< imb <<" nchannels:"<<image->channels()<<" step:"<<image->step<<endl;

    //ROW MAJOR


    for (int row = 0; row < image->rows; row++){

        for (int col = 0; col < image->cols; col++){

            imdata.push_back( (vl_sift_pix)(image->at<uchar>(row,col))/255.0 );
            //imdata.push_back( ((vl_sift_pix)ldata[j*image->step+i]) );
            //cout<<imdata[row*image->cols+col]<<" ";
        }
        //cout<<endl;
        //getchar();
    }



    //COL MAJOR
    /*

    for (int col = 0; col < image->cols; col++){
        for (int row = 0; row < image->rows; row++){
            imdata.push_back( (vl_sift_pix)(image->at<uchar>(row,col))/255.0 );
            //imdata.push_back( (vl_sift_pix)(image->at<uchar>(row,col)));
        }
        //cout<<endl;
        //getchar();
    }
    */

    //VlSiftFilt *filt = vl_sift_new(w , h , -1 , 3 , 0); //TEST
    VlSiftFilt *filt = vl_sift_new(image->cols , image->rows , noct , nlvl , omin); //inicializar filtro SIFT


    //cout << "DBG: procesando primera octava (0)..." << endl;
    //int retval = vl_sift_process_first_octave(filt , imdata); //procesar 1era octava
    int retval = vl_sift_process_first_octave(filt , &imdata[0]);

    while(1){ //mientras haya octavas que procesar

        if(VL_ERR_EOF == retval){ //procesar siguente octava
            break; //si no hay mas, terminar
        }

        vl_sift_detect(filt); //detectar keypoints
        //cout << "BOFGenerator::processImage(): encontrados " << (vl_sift_get_nkeypoints(filt)) <<" keypoints"<< endl;

        getDescriptors(filt,sstorage,siftKeypoints,&lastkp); //calcular descriptores
        //retval=VL_ERR_EOF; //forzar una sola octava
        retval = vl_sift_process_next_octave(filt);
    }



    cout << "BOFGenerator::processImage(): encontrados " << sstorage.size() <<" descriptores"<< endl;
    vl_sift_delete(filt); //destruir filtro para liberar memoria
    //delete[] imdata;
}

void BOFGenerator::flattenSIFTMatrix(const vector<vl_sift_pix*> &in, vector<vl_sift_pix> &out)
{

    //COL MAJOR
    /*
    for(int col=0; col<128; col++ ){ //transponer matriz
        for(int row=0; row<in.size(); row++ ){
            out.push_back( (in[row])[col] );
            //myfile << asifts[asifts.size()-1]<<",";
            if( (out[col*in.size()+row]>255) || (out[col*in.size()+row]<0) ){
                cout<<"ALERTA EN out["<<row<<"]["<<col<<"]: "<<out[col*in.size()+row]<<endl;
                getchar();
            }
        }
        //myfile << endl;
    }
    */

    //ROW MAJOR

    for(int row=0; row<in.size(); row++ ){ //transponer matriz

        for(int col=0; col<128; col++){

            out.push_back( (in[row])[col] );
            //myfile << asifts[asifts.size()-1]<<",";
            if( (out[row*128+col]>255) || (out[row*128+col]<0) ){

                cout<<"ALERTA EN out["<<row<<"]["<<col<<"]: "<<out[row*128+col]<<endl;
                getchar();
            }
        }
        //myfile << endl;
    }


}
