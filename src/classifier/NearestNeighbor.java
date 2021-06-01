package classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import util.Pair;
import weka.core.Instances;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 */
public class NearestNeighbor extends AbstractNearestNeighbor implements Serializable {
	private static final long serialVersionUID = 8662234558169046563L;

	protected double[] scaling;
	protected double[] translation;

	private static List<List<Object>> learnedData;
	List<Pair<List<Object>, Double>> listWithPairsBuffer;
	List<Pair<List<Object>, Double>> listWithPairsWithK;
	Pair<List<Object>, Double> pairInstanceDistance; 

	@Override
	public String getMatrikelNumbers() {
		String maNumbers = "2498827,2612917,2099682";
		return maNumbers;
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		learnedData = new LinkedList<List<Object>>();
		for(List<Object> o : data){
			learnedData.add(o);
		}
	}

	@Override
	protected List<Pair<List<Object>, Double>> getKNearest(List<Object> data) {
		listWithPairsBuffer = new LinkedList<Pair<List<Object>, Double>>();
		
		if(isNormalizing()) {
			double[][] result = normalizationScaling();
			scaling = new double[result[0].length];
			translation = new double[result[0].length];
			for (int i = 0; i < result[0].length; i++) {
				scaling[i] = result[0][i];
				translation[i] = result[1][i];
			}
			
			int i = 0;
			List<Object> normalizedData = new ArrayList<Object>();
			for (Object attribute: data) {

				if (attribute instanceof Double) {						
					normalizedData.add(((Double) attribute * scaling[i]) - translation[i]);
				}
				else {
					normalizedData.add(attribute);
				}
				i++;
			}
			data = normalizedData;
		}		
		
		for (List<Object> instance : learnedData) {
			List<Object> instance2 = instance;
			if (isNormalizing()) {
				List<Object> normalizedInstance = new ArrayList<Object>();

				int i = 0;
				for (Object attribute: instance) {
					if (attribute instanceof Double) {						
						normalizedInstance.add(((Double) attribute * scaling[i]) - translation[i]);
					}
					else {
						normalizedInstance.add(attribute);
					}
					i++;						
				}				
				instance = normalizedInstance;
			}

			double distanceBetween = 0;			
			if (getMetric()== 0) {
				distanceBetween = determineManhattanDistance(instance, data);
			}
			else {
				distanceBetween = determineEuclideanDistance(instance, data);
			}

			instance = instance2; //return unnormalized instance.
			pairInstanceDistance = new Pair<>(instance, distanceBetween);
			listWithPairsBuffer.add(pairInstanceDistance);
		}
		
		//Aufgabe 3.3 c)
		//Durch das Shuffeln vor der Sortierung ist garaniert, dass Instanzen mit gleichen Distanzen zufüllig geordnet sind
		//Collections.shuffle(listWithPairsBuffer); 
		
		listWithPairsBuffer.sort((p1, p2) -> p1.getB().compareTo(p2.getB()));

		listWithPairsWithK = new LinkedList<Pair<List<Object>, Double>>();
		for (int i=0; i < getK(); i++) {
			listWithPairsWithK.add(listWithPairsBuffer.get(i));
		}
		return listWithPairsWithK;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		double manhattanDistance = 0; 
		for(int i = 0; i < instance1.size(); i++) {
			double distanceValue= 0;
			if(i==getClassAttributeIndex()) {
				continue;
			}
			if( (instance1.get(i) instanceof String) && (instance2.get(i) instanceof String) ) {
				if(!Objects.equals(instance1.get(i), instance2.get(i))) {
					distanceValue = 1;
					manhattanDistance += distanceValue;
				}
			}
			else {
				Double Value1 = (Double) instance1.get(i);
				Double Value2 = (Double) instance2.get(i);
				distanceValue = Math.abs( (Value1 - Value2));
				manhattanDistance += distanceValue;
			}
		}
		return manhattanDistance;	
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		double euclideanDistance = 0; 
		for(int i = 0; i < instance1.size(); i++) {
			double distanceValue= 0;
			if(i==getClassAttributeIndex()) {
				continue;
			}
			if( (instance1.get(i) instanceof String) && (instance2.get(i) instanceof String) ) {
				if(!Objects.equals(instance1.get(i), instance2.get(i))) {
					distanceValue = 1;
					euclideanDistance += distanceValue;
				}
			}
			else {
				Double Value1 = (Double) instance1.get(i);
				Double Value2 = (Double) instance2.get(i);
				distanceValue = Math.abs( (Value1 - Value2));
				Double distanceValuePow = distanceValue*distanceValue;
				euclideanDistance += distanceValuePow;
			}
		}
		double resultEuclideanDistance = Math.sqrt(euclideanDistance);
		return resultEuclideanDistance;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> resultFromVotes; 

		if(isInverseWeighting()) {
			resultFromVotes = getWeightedVotes(subset);
		}
		else {
			resultFromVotes = getUnweightedVotes(subset);
		}

		Object resultFromGetWinner = getWinner(resultFromVotes);
		return resultFromGetWinner;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> weightedVotes = new HashMap<Object, Double>();
		for (Pair<List<Object>, Double> pair: subset) {
			Object currentClass = pair.getA().get(getClassAttributeIndex());
			double weight = 1 / pair.getB();
			if(!weightedVotes.containsKey(currentClass)) {
				weightedVotes.put(currentClass, weight);
			}
			else {
				Double currentVoteNumber = weightedVotes.get(currentClass);
				weightedVotes.put(currentClass, currentVoteNumber + weight);
			}
		}
		return weightedVotes;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> unweightedVotes = new HashMap<Object, Double>();
		for (Pair<List<Object>, Double> pair: subset) {
			Object currentClass = pair.getA().get(getClassAttributeIndex());
			if(!unweightedVotes.containsKey(currentClass)) {
				unweightedVotes.put(currentClass, 1.0);
			}
			else {
				Double currentVoteNumber = unweightedVotes.get(currentClass);
				currentVoteNumber++;
				unweightedVotes.put(currentClass, currentVoteNumber);
			}
		}
		return unweightedVotes;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		List<Entry<Object, Double>> winners = new ArrayList<Entry<Object, Double>>();

		for (Entry<Object, Double> entry : votes.entrySet()) {
			if (winners.size() == 0 || entry.getValue().compareTo(winners.get(0).getValue()) >= 0) {
				if (winners.size() == 0 || entry.getValue().compareTo(winners.get(0).getValue()) != 0) {
					winners.removeAll(winners);
					winners.add(entry);		
				}
				else {
					winners.add(entry);
				}
			}
		}
		Random r = new Random();
		return winners.get(r.nextInt(winners.size())).getKey();	
	}

	@Override
	protected double[][] normalizationScaling() {	
		Map<Integer, Double[]> attributesMinMax = new HashMap<Integer, Double[]>();
		for (List<Object> instance: learnedData) { //crawl learned data for min and max values of numeric attributes
			int i = 0;
			for (Object attribute: instance) {	
				if (attribute instanceof Double) { // Check if numeric attribute
					double value = (Double) attribute;
					if (!attributesMinMax.containsKey(i)) { // If there are no previous values, add as min and max
						attributesMinMax.put(i, new Double[]{value, value});
					}
					else {									// Check if new min/max is found
						Double[] minmax = attributesMinMax.get(i);
						if (minmax[0] > value) { 			// assign new min if necessary
							minmax[0] = value;
						}
						if (minmax[1] < value) {			// assign new max if necessary
							minmax[1] = value;
						}
						attributesMinMax.put(i, minmax);    // put minmax values back in map
					}
				}
				else {
					if (!attributesMinMax.containsKey(i)) { // If there is no entry generate one, for nominal with minmax values equaling null.
						attributesMinMax.put(i, null);
					}
				}
				i++;
			}
		}

		//use the minmax data to calculate scaling and translation for each attribute
		double[][] result = new double[2][attributesMinMax.size()];	
		for (int i: attributesMinMax.keySet()) {
			double scaling = 0;
			double translation = 0;
			if (attributesMinMax.get(i) != null) {
				Double[] minmax = attributesMinMax.get(i);	
				if (minmax[0].doubleValue() != minmax[1].doubleValue()) { 
					scaling = (1 / (minmax[1] - minmax[0]));				
					translation = minmax[0] / (minmax[1] - minmax[0]);
				}
			}
			result[0][i] = scaling;
			result[1][i] = translation;
		}
		return result;
	}
}
