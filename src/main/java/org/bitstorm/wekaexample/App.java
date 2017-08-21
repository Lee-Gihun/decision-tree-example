package org.bitstorm.wekaexample;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

/**
 * Implementation with Weka of the example from this
 * <a href="https://dzone.com/articles/machine-learning-with-decision-trees">article</a>
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
    	// load data from CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(App.class.getResourceAsStream("weather.csv"));
        Instances data = loader.getDataSet();
    	
        //initialize the info gain extractor
    	InfoGainAttributeEval eval = new InfoGainAttributeEval();
    	Ranker search = new Ranker();
    	
    	AttributeSelection attSelect = new AttributeSelection();
    	attSelect.setEvaluator(eval);
    	attSelect.setSearch(search);
    	attSelect.SelectAttributes(data);
    	
    	//let's show the information gain value for each attribute
    	System.out.println(attSelect.toResultsString());

    	//now we build and show the decision tree
    	Id3 tree = new Id3();
    	tree.buildClassifier(data);
    	
    	System.out.println(tree);
    }
}
