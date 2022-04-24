from scipy.optimize import linear_sum_assignment

from task_2 import get_similarity_matrix


def imperfect_match(root_path):
    similarity, video_name_list, audio_name_list = get_similarity_matrix(root_path)
    row_ind, col_ind = linear_sum_assignment(-similarity)

    results = {}

    for i in range(len(col_ind)):
        video_name = video_name_list[row_ind[i]]

        results[audio_name_list[col_ind[i]]] = int(video_name.split("_")[-1])

    for audio_name in audio_name_list:
        if audio_name not in results.keys():
            results[audio_name] = -1

    return results


if __name__ == "__main__":
    results = imperfect_match(root_path="../data/original_data/task3/test/0/")
    print(results)
