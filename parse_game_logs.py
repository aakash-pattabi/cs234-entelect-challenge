import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')



def generate_graphs(log_file):
    game_counter = 0
    games_won = [0]
    our_scores = [0]
    opp_scores = [0]

    with open(log_file) as f:
        content = f.readlines() 
        win_counter = 0
        game_counter = 0
        for i, l in enumerate(content):
            if game_counter==100: 
                break
            if "winner" in l:
                game_counter+=1
                if "A" in l:
                    win_counter+=1
                search_string = r'Score=\s+([0-9]*)'
                our_game_score=int(re.search(search_string, content[i-12]).group(1))
                opp_game_score=int(re.search(search_string, content[i-11]).group(1))

                games_won.append(win_counter)

                our_scores.append(our_game_score)
                opp_scores.append(opp_game_score)

        print(len(games_won))
        print(games_won)
        print(np.mean(our_scores))

    plt.plot(range(0, len(games_won)), games_won)    
    plt.savefig(log_file[log_file.rfind("/")+1:log_file.rfind(".")]+"_"+"win_graph.pdf")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type = str)
    args = parser.parse_args()
    generate_graphs(args.log_file)

if __name__ == "__main__":
    main()
