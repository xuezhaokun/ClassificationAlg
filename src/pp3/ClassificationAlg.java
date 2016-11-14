package pp3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
	
	public static int sigmoidPredict(double a) {
		double sigmoid = 1 / (1 + Math.exp(-a));
		if (sigmoid >= 0.5) {
			return 1;
		} else {
			return 0;
		}
	}
	
	public static void writeDataToFile(HashMap<Integer, double[]> error_statics, String outputFile) throws Exception {
		PrintWriter writer  = new PrintWriter(outputFile, "UTF-8");
		for (Map.Entry<Integer, double[]> entry : error_statics.entrySet()) {
			int key = entry.getKey();
			double[] statics = entry.getValue();
			writer.println(key + "," + statics[0] + "," + statics[1]);
		}
		writer.close();
	}
	
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
//		String dataPath = "data/";
//		String haha_data = dataPath + "haha.csv";
//		String haha_label = dataPath + "haha-labels.csv";
//		double[][] haha_dataset = ClassificationAlg.readData(haha_data);
//		double[] haha_labels = ClassificationAlg.readLabels(haha_label);
//		double[][] data_with_labels = ClassificationAlg.combineDataWithLabels(haha_dataset, haha_labels);
//		double[][] shuffled_data = ClassificationAlg.shuffleData(data_with_labels);
//		int data_size = shuffled_data.length;
//		int testing_size = data_size / 3;
//		int training_size = data_size - testing_size;
//		System.out.println("testing size " + testing_size + " training_size " + training_size);
//		List<double[][]> split_results = ClassificationAlg.splitTestAndTrain(shuffled_data, testing_size);
//		System.out.println(Arrays.deepToString(split_results.get(0)));
//		System.out.println("-------------");
//		System.out.println(Arrays.deepToString(split_results.get(1)));
//		System.out.println("=============");
//		double[][] pure_data = ClassificationAlg.getDataFromDataWithLabels(data_with_labels);
//		double[] pure_labels = ClassificationAlg.getLabelsFromDataWithLabels(data_with_labels);
//		System.out.println(Arrays.deepToString(pure_data));
//		System.out.println(Arrays.toString(pure_labels));
//		System.out.println("~~~~~~~~~~~~");
//
//		List<double[][]> data_by_class = GenerativeAlg.splitDataByClass(data_with_labels);
//		System.out.println(Arrays.deepToString(data_by_class.get(0)));
//		System.out.println(Arrays.deepToString(data_by_class.get(1)));
//		List<Double> ns = GenerativeAlg.countNs(data_by_class);
//		System.out.println(Arrays.deepToString(ns.toArray()));
//		System.out.println("++++++++++++++");
//		List<Matrix> mus = GenerativeAlg.calculateMus(data_by_class);
//		Matrix diff = mus.get(0).minus(mus.get(1));
//		System.out.println(Arrays.toString(mus.get(0).getColumnPackedCopy()));
//		System.out.println(Arrays.toString(mus.get(1).getColumnPackedCopy()));
//		System.out.println("~~~~~~~~~~~~~");
//		System.out.println(Arrays.toString(diff.getColumnPackedCopy()));
//		System.out.println("--------------");
//		Matrix s = GenerativeAlg.calculateS(data_by_class);
//		System.out.println(Arrays.deepToString(s.getArray()));
//		System.out.println("col: " + s.getColumnDimension() + " row: " + s.getRowDimension());
//		System.out.println(Arrays.deepToString(s.inverse().getArray()));
//		Matrix w = GenerativeAlg.calculateW(s, mus);
//		System.out.println(Arrays.deepToString(w.getArray()));
//		Matrix s = GenerativeAlg.calculateS(data_by_class);
//		System.out.println(Arrays.deepToString(s.getArray()));
		Task1.runTask1();
	}
}
