import sys
import shutil
#selects best model out of list and copies to output/data_appendix_output
file_list = [
["../../data/intermediate_data/RandomForest","RandomForest"],
["../../data/intermediate_data/Linear","Linear"],
["../../data/intermediate_data/Ridge","Ridge"],
["../../data/intermediate_data/Lasso","Lasso"],
["../../data/intermediate_data/GradientBoost","GradientBoost"],
["../../data/intermediate_data/SupportVector","SupportVector"],
["../../data/intermediate_data/MultilayerPerceptron","MultilayerPerceptron"],
["../../data/intermediate_data/kNearestNeighbor","kNearestNeighbor"],
["../../data/intermediate_data/MovingAverage","MovingAverage"],
["../../data/intermediate_data/Poisson","Poisson"]]

def get_rmse(file_prefix):
    data_path = f"{file_prefix}_{sys.argv[1]}_model_data.txt"
    model_path = f"{file_prefix}_{sys.argv[1]}_model.txt"
    with open(data_path,"r") as f:
        data = f.readline()
        data = data.split()
        return float(data[0]),(data_path,model_path)

def main():
    first_file = get_rmse(file_list[0][0])
    min_rmse = first_file[0]
    file_path_data = first_file[1][0]
    file_path_model = first_file[1][1]
    for i in range(1,len(file_list)):
        return_value = get_rmse(file_list[i][0])
        if return_value[0] < min_rmse:
            min_rmse = return_value[0]
            file_path_data = return_value[1][0]
            file_path_model = return_value[1][1]
    shutil.copy(file_path_data,f"../../output/data_appendix_output/{sys.argv[1]}_model_data.txt")
    shutil.copy(file_path_model,f"../../output/data_appendix_output/{sys.argv[1]}_model.txt")

if __name__ == "__main__":
    main()
