package pp3;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import Jama.Matrix;

public class GenerativeAlg {
	
	public static double[][] readData(String data_file, String label_file) throws IOException {
		double[][] dataset = ClassificationAlg.readData(data_file);
		double[] labels = ClassificationAlg.readLabels(label_file);
		return ClassificationAlg.combineDataWithLabels(dataset, labels);
	}

	public static List<double[][]> splitDataByClass(double[][] data_with_labels) {
		int dimension = data_with_labels[0].length;
		List<double[][]> data_by_class = new ArrayList<double[][]>();
		List<double[]> data_in_c1 = new ArrayList<double[]>();
		List<double[]> data_in_c2 = new ArrayList<double[]>();
		for (double[] data_record : data_with_labels) {
			if ((int) data_record[dimension - 1] == 1) {
				data_in_c1.addAll(Arrays.asList(data_record));
			} else {
				data_in_c2.addAll(Arrays.asList(data_record));
			}
		}
		data_by_class.add(data_in_c1.toArray(new double[][]{}));
		data_by_class.add(data_in_c2.toArray(new double[][]{}));
		return data_by_class;
	}
	
	public static List<Double> countNs(List<double[][]> data_by_class) {
		List<Double> ns = new ArrayList<Double>();
		double n1 = data_by_class.get(0).length;
		double n2 = data_by_class.get(1).length;
		ns.add(n1);
		ns.add(n2);
		return ns;
	}
	
	public static List<Double> calculateMus(List<double[][]> data_by_class) {
		double[][] data_in_c1 = data_by_class.get(0);
		double[][] data_in_c2 = data_by_class.get(1);
		List<Double> mus = new ArrayList<Double>();
		List<Double> ns = countNs(data_by_class);
		int dimension = data_in_c1[0].length;
		double mu1 = 0;
		double mu2 = 0;
		
		for (double[] data_c1 : data_in_c1) {
			for (int i = 0; i < dimension - 1; i++) {
				mu1 += data_c1[i];
			}
		}
		for (double[] data_c2 : data_in_c2) {
			for (int i = 0; i < dimension - 1; i++) {
				mu2 += data_c2[i];
			}
		}
		
		mu1 = mu1 / (ns.get(0));
		mu2 = mu2 / (ns.get(1));
		mus.add(mu1);
		mus.add(mu2);
		return mus;
	}
	
	public static Matrix calculateS(List<double[][]> data_by_class) {
		double[][] data_in_c1 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(0));
		double[][] data_in_c2 = ClassificationAlg.getDataFromDataWithLabels(data_by_class.get(1));
		int dimension = data_in_c1[0].length;
		List<Double> mus = calculateMus(data_by_class);
		double mu1 = mus.get(0);
		double mu2 = mus.get(1);
		List<Double> ns = countNs(data_by_class);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		double n = n1 + n2;
		for (int i = 0; i < data_in_c1.length; i++) {
			for (int j = 0; j < dimension; j++) {
				data_in_c1[i][j] = data_in_c1[i][j] - mu1; 	
			}
		}
		for (int i = 0; i < data_in_c2.length; i++) {
			for (int j = 0; j < dimension; j++) {
				data_in_c2[i][j] = data_in_c2[i][j] - mu2; 	
			}
		}
		Matrix d1 = new Matrix(data_in_c1);
		System.out.println("````````````````");
		System.out.println(Arrays.deepToString(d1.getArray()));
		System.out.println("````````````````");
		Matrix d2 = new Matrix(data_in_c2);
		Matrix s1 = d1.transpose().times(d1).times(1/n1);
		Matrix s2 = d2.transpose().times(d2).times(1/n2);
		Matrix s = s1.times(n1/n).plus(s1.times(n2/n));
		return s;
	}
	
	public static double calculateW0(Matrix s, List<Double> mus, List<Double> ns) {
		double[][] s_inverse = s.inverse().getArray();
		double mu1 = mus.get(0);
		double mu2 = mus.get(1);
		double n1 = ns.get(0);
		double n2 = ns.get(1);
		double c1_part = 0;
		double c2_part = 0;
		double w0 = 0;
		for (double[] s_row : s_inverse) {
			for (double s_cell : s_row) {
				c1_part += mu1 * s_cell * mu1;
				c2_part += mu2 * s_cell * mu2;
			}
		}
		c1_part = c1_part * (double)(-1/2);
		c2_part = c2_part * (double)(1/2);
		w0 = c1_part + c2_part + Math.log(n1/n2);
		return w0;
	}
	
	public static Matrix calculateW(Matrix s, List<Double> mus) {
		double mu1 = mus.get(0);
		double mu2 = mus.get(1);
		double diff = mu1 - mu2;
		Matrix s_inverse = s.inverse();
		int dimension = s_inverse.getColumnDimension();
		Matrix diffMatrix = new Matrix(dimension, 1, diff);
		return s_inverse.times(diffMatrix);
	}
	
	
	public static double[] predict(double[][] data_with_labels) { 
		int data_size = data_with_labels.length;
		int testing_size = data_size / 3;
		int training_size = data_size - testing_size;
		for (int i = 0; i < 30; i++) {
			double[][] shuffled_data = ClassificationAlg.shuffleData(data_with_labels);
			List<double[][]> splitted_data = ClassificationAlg.splitTestAndTrain(shuffled_data, testing_size);
			double[][] training_data = splitted_data.get(0);
			double[][] testing_data_labels = splitted_data.get(1);
			double[][] testing_data = ClassificationAlg.getDataFromDataWithLabels(testing_data_labels);
			double[] testing_label = ClassificationAlg.getLabelsFromDataWithLabels(testing_data_labels);
			Matrix testing_matrix = new Matrix(testing_data);
			
			for (int n = 20; n < training_size; n = n + 20) {
				double errors = 0;
				double error_rate = 0;
				if (n > training_size) {
					n = training_size;
				}
				double[][] current_training_data = ClassificationAlg.getFirstNData(training_data, n);
				List<double[][]> splitted_class = splitDataByClass(current_training_data);
				List<Double> ns = countNs(splitted_class);
				List<Double> mus = calculateMus(splitted_class);
				double mu1 = mus.get(0);
				double mu2 = mus.get(1);
				Matrix s = calculateS(splitted_class);
				double w0 = calculateW0(s, mus, ns);
				Matrix w = s.inverse().times((mu1 - mu2));
				double[] as = w.transpose().times(testing_matrix).getRowPackedCopy();
				for (int m = 0; m < testing_label.length; m++) {
					double predict_class = ClassificationAlg.sigmoidPredict(as[m]);
					if (predict_class != testing_label[m]) {
						errors++;
					}
				}
				error_rate = errors/(double)(testing_label.length);
			}
		}
	}

}
