from pandas import read_csv

def import_data():
    df = read_csv("boston.csv")
    print(df.head(5))
    return df


def main():
    import_data()


if __name__=="__main__":
    main()