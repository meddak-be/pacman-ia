from matplotlib.pyplot import title, plot, show, ylabel, xlabel, fill_between, scatter, hist, subplots, legend

def column(matrix, i):
    return [row[i] for row in matrix]

data= [[ 0 ,  0 ],
[ 0 ,  0.0 ],
[ 1 ,  0.0 ],
[ 2 ,  0.21347919360000006 ],
[ 3 ,  0.47543187388682906 ],
[ 4 ,  0.4902998037829588 ],
[ 5 ,  0.4906812517911424 ],
[ 6 ,  0.49068395726564695 ],
[ 7 ,  0.49068396357601834 ],
[ 8 ,  0.49068396358124394 ],
[ 9 ,  0.49068396358124544 ],
[ 10 ,  0.49068396358124544 ]
]

x = column(data, 0)
y = column(data, 1)

plot(x, y)
ylabel("valeur état initiale (0,0)")
xlabel("valeur de k")
show()

# score = [-510.47, -514.47, -513.21, -442.44, -271.65, -269.81, -240.83, -119.70, -199.81, 
# -118.41, -178.23, -5.84, 43.83, 22.41, 123.53, 105.82, 215.88, 226.03,  226.05, 146.63]

# score2 = [-510.30,
# -513.50,
# -513.01,
# -423.28,
# -250.54,
# -261.87,
# -280.03,
# -230.19,
# -150.67,
# -149.64,
# -150.18,
# -36.27,
# 22.89,
# 94.24,
# 124.38,
# 206.23,
# 195.88,
# 186.22,
# 216.35,
# 185.71]

# score3 = [-509.47,
# -512.83,
# -462.15,
# -361.94,
# -291.12,
# -270.20,
# -311.62,
# -230.61,
# -99.81,
# -138.78,
# -119.72,
# -168.79,
# -26.18,
# -16.86,
# -27.56,
# 225.56,
# 286.78,
# 145.89,
# 226.12,
# 236.11]

# score4 = [-509.79,
# -512.67,
# -472.20,
# -371.86,
# -301.30,
# -199.96,
# -281.52,
# -150.01,
# -220.25,
# -100.01,
# -199.74,
# -169.57,
# 84.03,
# 94.29,
# 33.75,
# 93.90,
# 266.98,
# 236.61,
# 226.06,
# 226.69]

# score5 = [-510.24,
# -512.75,
# -503.97,
# -399.97,
# -273.13,
# -302.71,
# -239.93,
# -200.79,
# -149.32,
# -89.11,
# -87.85,
# -87.47,
# 24.32,
# 3.97,
# 185.06,
# 134.79,
# 183.78,
# 135.88,
# 185.88,
# 216.50]

# score6 = [-510.88,
# -514.21,
# -503.02,
# -412.21,
# -372.62,
# -229.25,
# -218.96,
# -169.67,
# -100.35,
# -149.58,
# -100.21,
# -77.28,
# 33.27,
# 32.92,
# 64.24,
# 133.83,
# 216.53,
# 275.33,
# 125.67,
# 286.54]

# score7 = [-509.76,
# -513.89,
# -502.88,
# -403.75,
# -239.49,
# -311.51,
# -169.37,
# -169.78,
# -209.82,
# -230.44,
# -36.82,
# 84.34,
# 62.95,
# 53.29,
# 105.01,
# 125.95,
# 135.61,
# 246.53,
# 156.37,
# 196.28,]

# score8 = [-510.65,
# -513.97,
# -514.08,
# -452.78,
# -321.79,
# -272.42,
# -169.68,
# -179.97,
# -108.07,
# -117.42,
# -67.21,
# -16.46,
# 75.04,
# 103.29,
# 103.69,
# 134.53,
# 225.47,
# 175.32,
# 277.18,
# 276.00]

# score9 = [-510.53,
# -512.08,
# -503.56,
# -322.03,
# -240.35,
# -281.13,
# -240.92,
# -250.40,
# -190.32,
# -178.99,
# -26.75,
# -16.54,
# 124.18,
# 33.63,
# 125.28,
# 237.83,
# 156.28,
# 226.51,
# 256.67,
# 165.81]

# score10 = [-509.82,
# -512.04,
# -482.79,
# -412.16,
# -262.02,
# -311.28,
# -231.11,
# -220.51,
# -221.28,
# -110.42,
# -230.63,
# -17.64,
# -36.58,
# 54.47,
# 114.30,
# 207.09,
# 137.10,
# 136.03,
# 145.78,
# 225.84]

# scores = [score, score2, score3, score4, score5, score6, score7, score8, score9, score10]

# meanScore = []
# minScore = []
# maxScore = []
# lastScore = []

# for i in range(len(score)):
#     mean = 0
#     minn = 10000
#     maxx = -10000
#     for j in range(10):
#         curr = scores[j][i]
#         mean+= curr
#         if curr > maxx:
#             maxx=curr
#         if curr<minn:
#             minn=curr
#         if i == len(score)-1:
#             lastScore.append(scores[j][i])
#     maxScore.append(maxx)
#     minScore.append(minn)
#     mean/=10
#     meanScore.append(mean)


# print(meanScore)
# print(minScore)
# print(maxScore)
# print(lastScore)
# x = [i for i in range(100, 2001, 100)]
# print(x)



# # plot(x, meanScore, label="Moyenne des scores")
# # xlabel("Nombre d'épisodes")
# # ylabel("Score")
# # fill_between(x, minScore, maxScore, facecolor='C0', alpha=0.4, label="Ecart type")
# # legend(loc='upper left')
# # title("Moyenne et écart type des scores")
# plot(x, score10)
# xlabel("Nombre d'épisodes")
# ylabel("Score")
# title("Distribution des scores lors du dernier entrainement")
# show()