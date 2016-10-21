def main():
    res = []
    with open("parsed-new.csv") as file:
        for line in file:
            if "Россия" in line or "Russian" in line:
                a = line.split(",")
                x, y = a[0], a[1].split("/")[-1]
                res.append((x, y))
    with open("result.csv", "w") as rfile:
        for pair in res:
            rfile.write("{};{}".format(pair[0], pair[1]))

if __name__ == "__main__":
    main()
