package pp3;

import java.util.ArrayList;
import java.util.List;

public class PerformanceStat {
	public static double calcMean(List<Double> errors) {
		  double sum = 0;
		  if(!errors.isEmpty()) {
		    for (double e : errors) {
		        sum += e;
		    }
		    return sum / errors.size();
		  }
		  return sum;
	}
	
	
	public static double calStD(List<Double> errors) {
        double mean = calcMean(errors);
        double temp = 0;
        for(double e : errors) {
    			temp += (e-mean)*(e-mean);
        }
       double variance = temp / errors.size();
       return Math.sqrt(variance);
	}
	
	public static List<Double> calMeanTime(List<List<Double>> update_times) {
		int n = update_times.get(0).size();
		List<Double> average_times = new ArrayList<Double>();
		for (int i = 0; i < n; i++) {
			double time_1 = update_times.get(0).get(i);
			double time_2 = update_times.get(1).get(i);
			double time_3 = update_times.get(2).get(i);
			double average = (time_1 + time_2 + time_3) / 3;
			average_times.add(average);
		}
		return average_times;
	}
}
