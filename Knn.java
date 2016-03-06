/*This class is interface in between of BellaDati and Weka machine learning KNN algorithm.
 *It uses KNN algorithm and user can use text to speech method to speak out the results.To
 *use this class from within the BellaDati Formula or Transformation script console, compiled code
 *wrapped into the .jar has to be recorded into belladati_installation_library/WEB-INF/lib. Into the
 *same location weka.jar has to be recorded.Afterwards your application server has to be restarted.
 *
 * Use methods that start with comment GENERATE DATA to generate sample data to test and use this class as standalone.
 */




package com.belladati.knndemo;
import java.util.*;

import weka.classifiers.*;
import weka.classifiers.lazy.IBk;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.Instances;
import java.io.*;
 
public class Knn {
	
	
	
    
    /* Method builds Weka Dataset structure - Instances = dataset filled with multiple Instance = row*/
    
    public Instances buildInstance(List<List<String>> instancesdata,List<String> classifierValue, boolean traininstance){
    	
    	        
    			ArrayList<Attribute> attributeList = new ArrayList<Attribute>(instancesdata.size());//Declare list for attributes
    			Instances dataset;//Weka Instances
    			Instance  inst_row; //Weka Instance 
    			Attribute classifier = null; //Weka Attribute
    			Set<String> set = new HashSet<String>(classifierValue);
    			List<String> classifierUnique = new ArrayList<String>(set);//unique values of classifier
    			 
    			
    			// Declare all numeric attributes
    			for(int x = 0; x < instancesdata.size(); x++)
    			 {
    			     attributeList.add(new Attribute(String.valueOf(x)));
    			 }
    			
    			// Add nominal attribute/classifier along with its values
    			 
    			 classifier = new Attribute("classifier", classifierUnique);
    			 attributeList.add(classifier);
    			
    			 
    			 dataset = new Instances("Instances",attributeList,0);//Create Instances(dataset)	 
    			 dataset.setClassIndex(dataset.numAttributes() - 1); //Last attribute in dataset is classifier
    		
    			//Set value of attributes in each instance (row of dataset)
    			  for(int a = 0; a < instancesdata.get(0).size(); a++){
    				  
    		     inst_row = new DenseInstance(dataset.numAttributes()); // add new row into Instance (dataset)
    		     
    		     for(int x = 0; x < instancesdata.size(); x++)
    			 {
    			 inst_row.setValue(attributeList.get(x),Double.parseDouble(instancesdata.get(x).get(a))); // Populate each row(Instance) of Instances with value
    			 
    			 if(traininstance == true){
    				 inst_row.setValue(classifier,classifierValue.get(a));}
    			 }
    		     
    		     dataset.add(inst_row);
    		   
    			 }
    			  
    	return new Instances(dataset);// Our Instances with attributes,values ready for training or for classification
    }
    
    
/* KNN algorithm method.First argument takes data to train, second data to be classified, third class options
 * ,forth number of nearest neighbours to be compared with target instance to be classified*/	
    
 public static List<String> kNN(List<List<String>> train, List<List<String>> test,List<String> classifier ,int k) throws Exception {
	 
	    
	    Knn knn = new Knn();
	    Instances traininstance = knn.buildInstance(train,classifier, true); //Classifier values and also unique classifier values calculated
	    Instances testinstances = knn.buildInstance(test, classifier, false);//Only Unique classsifier values calculated
	    ArrayList<String> labels =  new ArrayList<String>(testinstances.numInstances());
	    
	    
	    IBk ibk = new IBk(k);		
		ibk.buildClassifier(traininstance);
		
		//To test the output when running outside BellaDati
		//System.out.println(testinstances.toSummaryString());
		
		for (int i = 0; i < testinstances.numInstances(); i++) {
			   double clsLabel = ibk.classifyInstance(testinstances.instance(i));
			   testinstances.instance(i).setClassValue(clsLabel);
			   labels.add(testinstances.classAttribute().value((int) clsLabel));
			   //System.out.println(labels.get(i));
			 }
	   
		 System.out.println(testinstances);
		//To test the output when running outside BellaDati
		 //System.out.println(testinstances.toSummaryString());
		
		//Evaluating trained model		
		 Evaluation eTest = new Evaluation(traininstance);
		 eTest.evaluateModel(ibk, testinstances);
		 //String strSummary = eTest.toSummaryString();
		 //System.out.println(strSummary);
			 
		 return (new ArrayList<String>(labels));
 }
	 
/*GENERATE DATA Random strings for classifier again when testing this class as standalone. Not inside BellaDati*/	 
	 public  String randomString(){
		 
		 Random rand = new Random();
		 int randomNum = rand.nextInt(3);
		 String[] strings = {"yellow","silver","black"};
		 String string = strings[randomNum];
		 return string;
	 }
	 
/*GENERATE DATA TrainData data generation. When testing this class as standalone*/
	 
	 public static List<List<String>> generateTrainData(int numofattributes, int numofraws, List<String> classifierValue ){

		 
		 //Training data set
		 List<List<String>> trainDataset = new ArrayList<List<String>>();
		 ArrayList<String> column = new ArrayList<String>();
		 
		 
		 //Classifier
		 classifierValue = new ArrayList<String>();
		 
		
		 
		 for(int a = 0; a < numofattributes; a++){
		 
		 for(int x = 0; x < numofraws; x++){
			 column.add(String.valueOf(Math.round(Math.random()*100)));
			 
		 }
			 trainDataset.add(new ArrayList<String>(column));
			 
			 column.clear();
			 
		}
	
		//Printing Traindataset when testing this class as standalone 
		 /*
		 for (int i = 0; i < trainDataset.size(); i++) {
				System.out.println(trainDataset.get(i));
			}
			 
			 /*for(int x = 0; x < trainDataset.size(); x++)
			 {
			     for(int z = 0; z < trainDataset.get(x).size(); z++)
			         {
			         System.out.print(trainDataset.get(x).get(z) + " ");
			         
			     }
			     System.out.println("");
			 }*/
		 
		 return trainDataset;
		 
			}
		 
	 
	 
	 /*GENERATE DATA ClassifierData for testing this class as standalone*/

	 public static ArrayList<String> generateClassifierData(int numofrows){
		 Knn knn = new Knn();
		 ArrayList<String> classifierValue = new ArrayList<String>(numofrows);
		 for(int x = 0; x < numofrows; x++){
			 classifierValue.add(knn.randomString()); 
			 
		 }
		 return classifierValue;
		 
	 }
		 
		 
		 
		
	 
	 
	/*GENERATE DATA testData using results of generateTrainData method without classifier column and selected num. of rows.*/
		 
	public static List<List<String>> generateTestData(int numofraws, List<List<String>> traindataset){
		 
		 List<List<String>> copy = new ArrayList<List<String>> (traindataset.size());
		 
		 //copyrows of train dataset
		 
for(int a = 0; a < traindataset.size(); a++){
				copy.add(new ArrayList<String>(traindataset.get(a)));
			 }
				 
		
		//remove rows of train dataset
		 
for(int a = 0; a < copy.size(); a++){
			 
			 for(int x = 0; x < numofraws; x++){
				 copy.get(a).remove(x);
				 
			 }
				 
			}

		 
		 
	 
		 
		 return copy;
		 
	 }
	
	
	
	 
	  /* This methods works only on MacOSX and sends the "say" command to the terminal with the appropriate args*/
	
	  static void saymac(String script, int speed, String voice) {
		  if (voice == "us" ){ voice = "Samantha";}
		  if (voice == "china" ){ voice = "Ting-ting";}
		  if (voice == "korea"){ voice = "Yuna";}
		  if (voice == "japan"){ voice = "Kyoko";} 
		  else {
			  voice = "Samantha";
		  }
		  
		  
	    try {
	      Runtime.getRuntime().exec(new String[] {"say", "-v", voice, "[[rate " + speed + "]]" + script});
	    }
	    catch (IOException e) {
	      System.err.println("IOException");
	    }
	  }
	 
	  // Overload the say method so we can call it with fewer arguments and basic defaults
	  static void saymac(String script) {
	    // 200 seems like a reasonable default speed
	    saymac(script, 200,"Samantha");
	  }
	 
	

	/*Use main method when running this class outside BellaDati*/
		 
	 public static void main(String[] args) throws Exception{
	
		 List<String> classifier = generateClassifierData(10);
		 List<List<String>> traindata = generateTrainData(3, 10,classifier);
		 List<List<String>> testdata  = generateTestData(5, traindata);
		 System.out.println(kNN(traindata, testdata, classifier, 2));
         
         
     
	 
 }
	
	 
	 
	
}
