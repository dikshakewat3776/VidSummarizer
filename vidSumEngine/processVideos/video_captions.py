import os
from utils import evaluate_beam_search


def get_captions(path):
  results = {}
  x = [x[0] for x in os.walk(path)]
  print(x)
  for dir in x[1:]:
      files = os.listdir(dir)
      for file in files:
        result, _ = evaluate_beam_search(dir + '/' + file, beam_index=5)
        print("result:" + str(result))
        result.pop(0)
        if len(result) <= 10:
          results[dir + '\\' + file] = ' '.join(result)
  return results


if __name__ == '__main__':
    print(get_captions('/home/dikshakewat/Projects/VideoSumEngine/cluster_frames'))
