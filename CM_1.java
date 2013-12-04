package weka.filters.supervised.attribute;


import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Remove;


public class CM_1 extends SimpleBatchFilter{
	
	private static final long serialVersionUID = 1L;
	protected int m_folds;
	private int m_range;
	public String m_targetclass = "first occuring";
	private Map<String, Double> RankingSums;
	private Map<String, Double> rangedRankings;
	Map<String,List <Double>> GraphScores = new HashMap<String, List<Double>>();
	private boolean m_graphcomputed;
	
	
	public CM_1(){
		m_folds = 10;
		m_range = 10;
		
		RankingSums = new HashMap<String, Double>();
		rangedRankings = new HashMap<String, Double>();
		
	}

	
	public String globalInfo() {
	
		return "A supervised attribute filter is used to compute " 
		  + "the CM_1 score for different datasets . It uses a 10 fold cross validation"
		  + "and creates a CM_1 score for each fold finally combined to one big ranking.";
	}
	
	public Enumeration listOptions() {
//		
	    Vector<Option> result;
    	Enumeration enm;
//		
		result = new Vector<Option>();
	
	    enm = super.listOptions();
	    while (enm.hasMoreElements())
	      result.addElement((Option) enm.nextElement());
	
	    result.addElement(new Option("\tThe number of folds (default: 10).\n",
	        "-f", 1, "-f <int>"));
	    
	    result.addElement(new Option("\tThe number of elements taken for ranking (default: 5).\n",
		        "-R", 1, "-R <int>"));
	    
	    result.addElement(new Option("\tThe target Class (default: 1).\n",
		        "-C", 1, "-C <int>"));

	    
	    return result.elements();
	  }
	  
	public void setOptions(String[] options) throws Exception {
			
		 String numberofFoldsString = Utils.getOption('k', options);
			if (numberofFoldsString.length() != 0) {
			  setNumberofFolds((Integer.parseInt(numberofFoldsString)));
			} else {
				setNumberofFolds(10);
			} 
		    
	    String rangeString = Utils.getOption('R', options);
	    if (rangeString.length() != 0) {
	      setRange((Integer.parseInt(rangeString)));
	    } else {
	    	setRange(5);
	    }
	    
	    String targetClassString = Utils.getOption('C', options);
	    if (targetClassString.length() != 0) {
		      setTargetClass(targetClassString);
		    }
	    else {
	    	setTargetClass("first occuring");
	    }
	    String graphComputedString = Utils.getOption('G', options);
	    if (graphComputedString.length() != 0) {
		      setGraphComputed(Boolean.parseBoolean(targetClassString));
		    }
	    else {
	    	setGraphComputed(false);
	    }
	    
		    
		    if (getInputFormat() != null)
		        setInputFormat(getInputFormat());
	  }


	public String[] getOptions(){
		 Vector<String> result = new Vector<String>();
	    String[] options = super.getOptions();
	    for (int i = 0; i < options.length; i++) {
	      result.add(options[i]);
	    }

	    result.add("-k");
	    result.add("" + getNumberofFolds());
	    result.add("-T");
	    result.add("" + getRange());
	    result.add("-C");
	    result.add("" + getTargetClass());
	    result.add("-G");
	    result.add("" + isGraphComputed());
	    
	    
	    
	    return result.toArray(new String[result.size()]);
	}
	public boolean isGraphComputed() {
		
		return m_graphcomputed;
	}
	
	public void setGraphComputed(boolean graphcomputed){
		m_graphcomputed = graphcomputed;
	}
	
	public String graphcomputedTipText(){
		return "Specifies if a optional graph of the results should be computed";
	}
	
	public int getNumberofFolds() {
		return m_folds;
	}
	
	public void setNumberofFolds(int folds) {
		m_folds = folds;
	  }
	
	public String numberofFoldsTipText(){
		return "The Number of Folds which are used for Cross-Validation";
	}
	
	public int getRange() {
		return m_range;
	}
	
	public void setRange(int range) {
		m_range = range;
		
	}
	
	public String RangeTipText(){
		return "The Number of attributes taken for Ranking";
	}

	public String getTargetClass(){
		return m_targetclass;
	}
	
	public void setTargetClass(String targetclass) {
		m_targetclass = targetclass;
	  }
	
	public String targetClassTipText(){
		return "Sets the target class for the attribute selection as an numerical value";
	}
	
	public Capabilities getCapabilities(){
		Capabilities result = super.getCapabilities();
		result.enableAllAttributes();
		result.enableAllClasses(); //// filter doesn't need class to be set//
	    return result;
	}
	
	protected Instances determineOutputFormat(Instances inputFormat) {
		 Instances result = new Instances(inputFormat);	//output format same like input without any new attributes, all instances copied
		 return result;
	 }
	 
	protected Instances process(Instances inst) throws Exception {
		Instances result = new Instances(determineOutputFormat(inst), 0);
	    for (int i = 0; i < inst.numInstances(); i++) {
	       double[] values = new double[result.numAttributes()];
	       for (int n = 0; n < inst.numAttributes(); n++)
	         values[n] = inst.instance(i).value(n);
	       result.add(new DenseInstance(1, values));
	     }
	     setInputFormat(result);
	     createFolds(result);
	     
	     
	     applyRangesandComputeJSON(RankingSums);
	     
	     Instances finalresult = adjustInstances(result);
	     setOutputFormat(finalresult);
	     return finalresult;
	 }

	public void createFolds(Instances inputdata) throws Exception{
		 StratifiedRemoveFolds remove = new StratifiedRemoveFolds();
		 remove.setNumFolds(m_folds); 
		 remove.setSeed(1);
		 remove.setInvertSelection(true);									//divide in n-folds and invert to get only one fold at a time
		 
		 for(int i = 1; i <= m_folds; i++){
			 remove.setFold(i);
			 remove.setInputFormat(inputdata); 
			 Instances modifiedData = Filter.useFilter(inputdata, remove);
			 
			 computeCM1(modifiedData);
		 }
		 //compute_RankingwithAverages();
	 }
	 
	public void computeCM1(Instances fold) throws IOException{
		 
		 int numAttributes = fold.numAttributes();
		 int num_instances = fold.numInstances();
		 
		 Map<String, Double> CM_1Scores = new HashMap<String, Double>();    //create a HashMap to store the CM_1 Score for each fold
		 
		 for(int attribute = 0; attribute < numAttributes-1; attribute++){		//get sum for each attribute column
			 double sum_specificClass = 0.0;											
			 double sum_otherClasses = 0.0;
			 int num_specificClass = 0;
			 int num_otherClasses = 0;
			 double min = -1.000;														//initialize with -1 making the assumption that there are only positive values in the dataset
			 double max = 0.000;
					 
			 for(int instance = 0 ; instance < num_instances; instance++){			//for each instance get the data
				 Instance inst = fold.get(instance);
				 double value = inst.value(attribute);
				 
				 
				 if(m_targetclass.equals("first occuring")){
					 m_targetclass = inst.classAttribute().value(0);
				 }//if no class was choosen the first occuring is choosen
				 
				 if((inst.classAttribute().value((int)inst.classValue())).equals(m_targetclass)){
					 sum_specificClass += value;										//add attributes value to sum array
					 num_specificClass +=1;

				 }
				 else{
					 sum_otherClasses += value;
				 	 num_otherClasses  += 1;

				 	 
				 	 if(value < min || min == -1)											//calculate max and min value
				 		 min = value;
				 	 else if (value > max)
				 		 max = value;
				 	}
				 
			 	}
			 
			 double over = ((1.0/num_specificClass)*sum_specificClass) - (sum_otherClasses*(1.0/num_otherClasses));
			 double under = 1+(max-min);

			 double CM_1Score = over/under;
			 
			 if(GraphScores.get(fold.attribute(attribute).name())==null){	
				 GraphScores.put(fold.attribute(attribute).name(),  new ArrayList<Double>());
			 }
			 
			 GraphScores.get(fold.attribute(attribute).name()).add(CM_1Score); // put CM_1 score for each column of attribute
			 
			 CM_1Scores.put(fold.attribute(attribute).name(), Math.abs(CM_1Score)); // put absolute CM1 Score


	 		} //all attributes computed
		 compute_Ranking(CM_1Scores);	//compute Ranking with absolute values if computation if average CM1 Scores needed run compute_RankingwithAverages after all createFolds()
		 }
	 
	public void compute_Ranking(Map<String, Double> abs_CM_1Scores){	
		Map<String, Double> sorted = sortByValues(abs_CM_1Scores);
		List<String> sortedAsArray = new ArrayList<String>(sorted.keySet());		//convert Keys to array
		 
		 for(int i = sortedAsArray.size(); i > 0; i--){
			 if(!RankingSums.containsKey(sortedAsArray.get(i-1))){
				 RankingSums.put(sortedAsArray.get(i-1), 1.0 * sortedAsArray.size() - i + 1);
			 }
			 else{
				 RankingSums.put(sortedAsArray.get(i-1), RankingSums.get(sortedAsArray.get(i-1)) + (1.0 * sortedAsArray.size() - i + 1)) ;	//update attributes ranking sum; ranking[attribute] = previous sum + index
			 }
		 }
	 }
	
	public void compute_RankingwithAverages(){
		Map<String, Double> Scores = new HashMap<String, Double>();
		
		for (Map.Entry pairs : GraphScores.entrySet()) {
			Scores.put(pairs.getKey().toString(), calculateAverage(GraphScores.get(pairs.getKey())));
		}
		Map<String, Double> sorted = sortByValues(Scores);
		List<String> sortedAsArray = new ArrayList<String>(sorted.keySet());		//convert Keys to array
		
		for(int i = 0; i< sortedAsArray.size(); i++){
				 RankingSums.put(sortedAsArray.get(i), sorted.get(sortedAsArray.get(i)));
		}
	}
	 
	public Instances adjustInstances( Instances input) throws Exception{
		 String IndicesToBeRemoved = "";
		 for (int i = 0; i < input.numAttributes(); i++) {
		      Attribute att = input.attribute(i);
		      if(!(rangedRankings.containsKey(att.name())) && att.index()!= input.classIndex()){
		    	  IndicesToBeRemoved += String.valueOf(i+1) + ",";
		      }
		 }
		 if(!IndicesToBeRemoved.isEmpty()){
			 IndicesToBeRemoved = IndicesToBeRemoved.substring(0, IndicesToBeRemoved.length() - 1); //remove additional comma at end
			 Remove remove = new Remove();
			 remove.setAttributeIndices(IndicesToBeRemoved);
		   	 remove.setInputFormat(input);
		   	 input = Filter.useFilter(input, remove);
		 }
		 
		 return input;
	}
		 
	public void applyRangesandComputeJSON (Map<String, Double> sortedbyRanking) throws IOException{
		
		 sortedbyRanking = sortByValues(sortedbyRanking);
		 int index = 1;
		 String jsontopattributes = "{\"key\": \"topattributes\", \"color\": \"#d62728\"  , \"values\": [";
		 String jsonleastattributes = "[{\"key\": \"bottomattributes\", \"color\": \"#2ca02c\",  \"values\": [";
		 String jsonmiddleattributes = "{\"key\": \"middleattributes\", \"color\": \"#1f77b4\",  \"values\": [";
		 
		 
		 for(Map.Entry pairs : sortedbyRanking.entrySet()){
			 if(index <= m_range && calculateAverage(GraphScores.get(pairs.getKey())) <= 0.0 ){
				 jsonleastattributes = jsonleastattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  String.valueOf(calculateAverage(GraphScores.get(pairs.getKey()))) + "} , ";	
				 rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
			 }
			 else if(index <= m_range && calculateAverage(GraphScores.get(pairs.getKey())) > 0.0 ){
				 jsontopattributes = jsontopattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  String.valueOf(calculateAverage(GraphScores.get(pairs.getKey()))) + "} , ";	
				 rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
			 }
			 else if((sortedbyRanking.size() - m_range > 1000 && index % 10 ==0 ) || (index > m_range) && sortedbyRanking.size() - m_range < 1000){
				 	jsonmiddleattributes = jsonmiddleattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  String.valueOf(calculateAverage(GraphScores.get(pairs.getKey()))) + "} , ";	
				 }
			 index++;
		 }
		 
		 if(jsontopattributes.length() > 68)
			 jsontopattributes = jsontopattributes.substring(0, jsontopattributes.length() - 2) + "]}]"; //remove additional comma at end
		 else
			 jsontopattributes = jsontopattributes + "]}]";
			 
		 if(jsonleastattributes.length() > 68)
			 jsonleastattributes = jsonleastattributes.substring(0, jsonleastattributes.length() - 2) + "]},"; //remove additional comma at end
		 else
			 jsonleastattributes = jsonleastattributes + "]},";
		 
		 if(jsonmiddleattributes.length() > 68)
		 jsonmiddleattributes = jsonmiddleattributes.substring(0, jsonmiddleattributes.length() - 2) + "]},"; //remove additional comma at end
		 else
			 jsonmiddleattributes = jsonmiddleattributes + "]},";
		  
		  //use Json to compute Graph if user set the option
		  if(isGraphComputed()){
			  String finaljson = "CM_1data = " +jsonleastattributes + jsonmiddleattributes+ jsontopattributes;
			  try {
				  generateGraph(finaljson);
			  } catch (IOException e) {
					e.printStackTrace();
			  }
		  }
			  
			  
				  
		  }
	
	private void generateGraph(String finaljson) throws IOException {
			  
		 FileWriter file = new FileWriter("CM1.json");
		 file.write(finaljson);
		 file.flush();
		 file.close();
		
		 String operatingSystem = System.getProperty("os.name");
		  
		 if(operatingSystem.startsWith("Windows"))
			 Runtime.getRuntime().exec("cmd start CM1.html");
		 if(operatingSystem.startsWith("Mac"))
			 Runtime.getRuntime().exec("open CM1.html");
		 if(operatingSystem.startsWith("Linux")){
			 Runtime.getRuntime().exec("xdg-open CM1.html");
		 }
		
	}

	public static <K extends Comparable,V extends Comparable> Map<K,V> sortByValues(Map<K,V> map){
	        List<Map.Entry<K,V>> entries = new LinkedList<Map.Entry<K,V>>(map.entrySet());
	      
	        Collections.sort(entries, new Comparator<Map.Entry<K,V>>() {
	
	            @Override
	            public int compare(Entry<K, V> o1, Entry<K, V> o2) {
	                return o1.getValue().compareTo(o2.getValue());
	            }
	        });
	      
	        //LinkedHashMap will keep the keys in the order they are inserted
	        //which is currently sorted on natural ordering
	        Map<K,V> sortedMap = new LinkedHashMap<K,V>();
	      
	        for(Map.Entry<K,V> entry: entries){
	            sortedMap.put(entry.getKey(), entry.getValue());
	        }
	      
	        return sortedMap;
	    }

	private double calculateAverage(List <Double> Scores) {
		double sum = 0;
		  if(!Scores.isEmpty()) {
		    for (double score : Scores) {
		        sum += score;
		    }
		    return sum / Scores.size();
		  }
		  return sum;
		}
	
	public static void main(String[] args) {
		 runFilter(new CM_1(), args);
	 }
	
	
	

}