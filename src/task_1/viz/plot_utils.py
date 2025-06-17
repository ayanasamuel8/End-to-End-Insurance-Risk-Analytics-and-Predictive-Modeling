def plot_top_bottom_claims(df, group_col, value_col, top_n=5):
    grouped = df.groupby(group_col)[value_col].mean().sort_values()
    top = grouped.tail(top_n)
    bottom = grouped.head(top_n)

    plt.figure(figsize=(12, 6))
    top.plot(kind="bar", color="green", label="Top")
    bottom.plot(kind="bar", color="red", label="Bottom")
    plt.title(f"Top and Bottom {top_n} {group_col}s by Average {value_col}")
    plt.ylabel(f"Avg {value_col}")
    plt.legend()
    plt.show()
