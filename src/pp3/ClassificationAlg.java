package pp3;

import java.io.BufferedReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import Jama.LUDecomposition;
import Jama.Matrix;

public class ClassificationAlg {
	/**
	 * The function parses an input file to a list of strings
	 * @param filename input file name
	 * @return a matrix
	 * @throws IOException input/output exception
	 */
	public static double[][] readData(String filename) throws IOException {

        List<double[]> examples = new ArrayList<double[]>();
        double[][] phi = null;
		// try to open and read the file
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
            String fileRead = br.readLine();
            // parse file content by space and add each token to the list
            while (fileRead != null) {
                String[] tokens = fileRead.split(",");
                double[] example = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                		example[i] = Double.parseDouble(tokens[i]);
                }
                examples.add(example);
                fileRead = br.readLine();
            }
            phi = new double[examples.size()][0];
            br.close();
            
        } catch (FileNotFoundException fnfe) {
            System.err.println("file not found");
        }
		return examples.toArray(phi);
	}
	
	/**
	 * The function to read label file
	 * @param filename label file name
	 * @return an array contains all labels
	 * @throws IOException
	 */
	public static double[] readLabels(String filename) throws IOException {

        List<Double> labels = new ArrayList<Double>();
        double[] t = null;
		// try to open and read the file
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
            String fileRead = br.readLine();
            // parse file content by space and add each token to the list
            while (fileRead != null) {
            		labels.add(Double.parseDouble(fileRead));
                fileRead = br.readLine();
            }
            t = new double[labels.size()];
            for (int i = 0; i < t.length; i++) {
                t[i] = labels.get(i);
             }
            br.close();
            
        } catch (FileNotFoundException fnfe) {
            System.err.println("file not found");
        }
		return t;
	}
	
	/**
	 * Combine data with labels
	 * @param data the data matrix
	 * @param labels the label vector
	 * @return a matrix by adding the label vector as the last column to the data matrix
	 */
	public static double[][] combineDataWithLabels(double[][] data, double[] labels) {
		int dimension = data[0].length + 1;
		double[][] data_with_labels = new double[data.length][dimension];
		for (int i = 0; i < data.length; i++) {
			double[] data_label = new double [dimension];
			for (int j = 0; j < data[i].length; j++) {
				data_label[j] = data[i][j];
			}
			data_label[dimension - 1] = labels[i];
			data_with_labels[i] = data_label;
		}
		return data_with_labels;
	}
	
	public static double[][] combineData(String data_file, String label_file) throws IOException {
		double[][] dataset = ClassificationAlg.readData(data_file);
		double[] labels = ClassificationAlg.readLabels(label_file);
		return ClassificationAlg.combineDataWithLabels(dataset, labels);
	}
	
	/**
	 * Extract data from data with labels matrix
	 * @param data_with_labels a matrix combined by data and labels
	 * @return the data matrix from the data_with_labels array
	 */
	public static double[][] getDataFromDataWithLabels(double[][] data_with_labels) {
		int dimension = data_with_labels[0].length - 1;
		double[][] pure_data = new double[data_with_labels.length][dimension];
		
		for (int i = 0; i < data_with_labels.length; i++) {
			for (int j = 0; j < dimension; j++) {
				pure_data[i][j] = data_with_labels[i][j];
			}
		}
		return pure_data;
	}
	
	/**
	 * Extract labels from data with labels matrix
	 * @param data_with_labels a matrix combined by data and labels
	 * @return the labeles vector from the data_with_labels matrix
	 */
	public static double[] getLabelsFromDataWithLabels(double[][] data_with_labels) {
		double[] labels = new double[data_with_labels.length];
		int dimension = data_with_labels[0].length;
		for (int i = 0; i < data_with_labels.length; i++) {
			labels[i] = data_with_labels[i][dimension-1];
		}
		return labels;
	}
	
	/**
	 * get the first n data from the data matrix we have
	 * @param data the data matrix we have
	 * @param n the first data we want
	 * @return the first n data from the given data matrix
	 */
	public static double[][] getFirstNData(double[][] data, int n) {
		int dimention = data[0].length;
		double[][] results = new double[n][dimention];
		for (int i = 0; i < n; i++){
			results[i] = data[i];
		}
		return results;
	}
	
	public static List<double[][]> splitTestAndTrain(double[][] data, int n) {
		List<double[][]> split_reuslts = new ArrayList<double[][]>();
		int data_size = data.length;
		int dimension = data[0].length;
		double[][] testing_data = getFirstNData(data, n);
		double[][] training_data = new double[data_size-n][dimension];
		for (int i = n; i < data_size; i++) {
			int index = i - n;
			training_data[index] = data[i];
		}
		split_reuslts.add(training_data);
		split_reuslts.add(testing_data);
		return split_reuslts;
	}
	
	public static double[][] shuffleData(double[][] data_with_labels) {
	    List<double[]> data_records = new ArrayList<double[]>();
	    for (double[] data_record : data_with_labels) {
	    		data_records.addAll(Arrays.asList(data_record));
	    }
	    Collections.shuffle(data_records);
	    return data_records.toArray(new double[][]{});
	}
	
	public static double sigmoid(double a) {
		double result = 1 / (1 + Math.exp(-a));
		return result;
	}
	
	public static int sigmoidPredict(double a) {
		double result = sigmoid(a);
		if (result >= 0.5) {
			return 1;
		} else {
			return 0;
		}
	}
	
	public static Matrix ludecompForInvert(Matrix a) {
		int dimension = a.getColumnDimension();
		Matrix identity = Matrix.identity(dimension, dimension);
		LUDecomposition lua = a.lu();
		return lua.solve(identity);
	}
	
	public static double[][] addW0ToData(double[][] dataset){
		int dimension = dataset[0].length;
		int data_length = dataset.length;
		double[][] data_with_w0 = new double[data_length][dimension + 1];
		for (int i = 0; i < data_length; i++) {
			for (int j = 0; j < dimension; j++) {
				data_with_w0[i][j] = dataset[i][j];
			}
			data_with_w0[i][dimension] = (double) 1;
		}
		return data_with_w0;
	}
	
	public static void writeTask1ToFile(HashMap<Integer, double[]> error_statics, String outputFile) throws Exception {
		PrintWriter writer  = new PrintWriter(outputFile, "UTF-8");
		Map<Integer, double[]> sorted_results = new TreeMap<Integer, double[]>(error_statics);
		System.out.println("training size |       mean       |      standard deviation     |");
		System.out.println("______________________________________________________________");
		for (Map.Entry<Integer, double[]> entry : sorted_results.entrySet()) {
			int key = entry.getKey();
			double[] statics = entry.getValue();
			System.out.println(key + "           " + statics[0] + "    " + statics[1]);
			writer.println(key + "," + statics[0] + "," + statics[1]);
		}
		writer.close();
	}
	
	public static void writeTask2ToFile(List<Double> timestamps, List<Double> error_rates, String outputFile) throws Exception {
		PrintWriter writer  = new PrintWriter(outputFile, "UTF-8");
		System.out.println("     time       |      error rate     |");
		System.out.println("_______________________________________");
		for (int i = 0; i < timestamps.size(); i++) {
			System.out.println(timestamps.get(i) + "     " + error_rates.get(i));
			writer.println(timestamps.get(i) + ", " + error_rates.get(i));
		}
		writer.close();
	}
	
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
//		
//		long tStart = System.currentTimeMillis();
		System.out.println("******** Running Task 1 ********");
		System.out.println("For each dataset I use 10 different sizes training data");
		System.out.println("Each dataset repeats 30 times, totally 300 routines for each dataset.");
		System.out.println("It takes around 10 - 15 minutes to finish task 1");
		//Task1.runTask1();

		System.out.println("******** Running Task 2 ********");
		Task2.runTask2();
//		long tEnd = System.currentTimeMillis();
//		long tDelta = tEnd - tStart;
//		double elapsedSeconds = tDelta / 1000.0;
//		System.out.println("running time ... " + elapsedSeconds);
		//Task2.runTask2();
	}
}
