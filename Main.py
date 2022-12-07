# https://www.kaggle.com/datasets/whenamancodes/students-performance-in-exams

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection

df = pd.read_csv("exams.csv")

#df["gender"].value_counts().plot(kind="bar", rot=0)

# --------------------------------------<Convert values to ints for analysis>-------------------------------------------
df.replace(to_replace="male", value=1, inplace=True)
df.replace(to_replace="female", value=2, inplace=True)
df.replace(to_replace="group A", value=1, inplace=True)
df.replace(to_replace="group B", value=2, inplace=True)
df.replace(to_replace="group C", value=3, inplace=True)
df.replace(to_replace="group D", value=4, inplace=True)
df.replace(to_replace="group E", value=5, inplace=True)
df.replace(to_replace="some high school", value=1, inplace=True)
df.replace(to_replace="high school", value=2, inplace=True)
df.replace(to_replace="some college", value=3, inplace=True)
df.replace(to_replace="associate's degree", value=4, inplace=True)
df.replace(to_replace="bachelor's degree", value=5, inplace=True)
df.replace(to_replace="master's degree", value=6, inplace=True)
df.replace(to_replace="free/reduced", value=1, inplace=True)
df.replace(to_replace="standard", value=2, inplace=True)
df.replace(to_replace="completed", value=1, inplace=True)
df.replace(to_replace="none", value=2, inplace=True)
# --------------------------------------</Convert values to ints for analysis>------------------------------------------
#scatter_matrix(df[["writing score", "reading score", "math score"]])

for grade in range(60, 101):
    df["math score"].replace(to_replace=grade, value='pass', inplace=True)
    df["reading score"].replace(to_replace=grade, value='pass', inplace=True)
    df["writing score"].replace(to_replace=grade, value='pass', inplace=True)

for grade in range(0, 60):
    df["math score"].replace(to_replace=grade, value='fail', inplace=True)
    df["reading score"].replace(to_replace=grade, value='fail', inplace=True)
    df["writing score"].replace(to_replace=grade, value='fail', inplace=True)


lm = linear_model.LogisticRegression(max_iter=1000)
math = df.values[:, 5]
reading = df.values[:, 6]
writing = df.values[:, 7]
x = df.values[:, 0:5]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, reading, test_size=0.20)
lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

a, b, c, d, e = 0, 0, 0, 0, 0

print("Type the associated integer for your answers")

f = open("invalid.txt", "a")

while(a == 0):
    try:
        a = int(input("What is the gender?\n\t1. male\n\t2. female\n"))
    except ValueError:
        print("Integer needed")
        f.write("Non int value typed in for gender\n")
    if (a < 1 or a > 2):
        print("Select a valid option")
        a = 0

while(b == 0):
    try:
        b = int(input("What is the race/ethnicity?\n\t1. Group A\n\t2. Group B\n\t3. Group C\n\t4. Group D\n\t5. Group E\n"))
    except ValueError:
        print("Integer needed")
        f.write("Non int value typed in for race/ethnicity\n")
    if (b < 1 or b > 5):
        print("Select a valid option")
        b = 0

while(c == 0):
    try:
        c = int(input("What is the parental level of education?\n\t1. some high school\n\t2. high school\n\t3. some college\n\t"
            "4. associate's degree\n\t5. bachelor's degree\n\t6. master's degree\n"))
    except ValueError:
        print("Integer needed")
        f.write("Non int value typed in for parental level of education\n")
    if (c < 1 or c > 6):
        print("Select a valid option")
        c = 0

while(d == 0):
    try:
        d = int(input("How is your meal plan structured?\n\t1. free/reduced\n\t2. standard\n"))
    except ValueError:
        print("Integer needed")
        f.write("Non int value typed in for meal plan\n")
    if (d < 1 or d > 2):
        print("Select a valid option")
        d = 0

while(e == 0):
    try:
        e = int(input("Was the test preparation completed?\n\t1. Yes\n\t2. No\n"))
    except ValueError:
        print("Integer needed")
        f.write("Non int value typed in for test preparation\n")
    if (e < 1 or e > 2):
        print("Select a valid option")
        e = 0

f.close()

lm.fit(x,math)
print("For math, the student is more likely to", lm.predict([[a,b,c,d,e]])[0])
print("The probability of passing math is %s \n" % f'{100*lm.predict_proba([[a,b,c,d,e]])[0][1]:.2f}%')
lm.fit(x,reading)
print("For reading, the student is more likely to", lm.predict([[a,b,c,d,e]])[0])
print("The probability of passing reading is %s \n" % f'{100*lm.predict_proba([[a,b,c,d,e]])[0][1]:.2f}%')
lm.fit(x,writing)
print("For writing, the student is more likely to", lm.predict([[a,b,c,d,e]])[0])
print("The probability of passing writing is %s \n" % f'{100*lm.predict_proba([[a,b,c,d,e]])[0][1]:.2f}%')
