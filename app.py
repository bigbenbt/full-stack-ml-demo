from bokeh.models import ColumnDataSource
from flask import Flask, render_template, request
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.embed import components
from bokeh.plotting import figure
from data_importer import import_data
from numpy import histogram
from ml import random_forest

app = Flask(__name__)

df = import_data()
size = df.size


@app.route("/")
def landing():
    current_feature_name = request.args.get("feature_name")
    if current_feature_name is None:
        current_feature_name = "medv"
    feature_names = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv"]
    script1, div1 = components(create_histogram(current_feature_name, 10))
    script2, div2=components(create_summary_stats_table(current_feature_name))
    important_features = None
    third_script = None
    if (current_feature_name=="medv"):
        important_features, third_script = components(create_feature_importance_table(random_forest(current_feature_name)))
    return render_template("index.html",
                           div1=div1,
                           div2=div2,
                           div3=important_features,
                           feature_names=feature_names,
                           script1=script1,
                           script2=script2,
                           script3=third_script,
                           title1=("Summary of " + current_feature_name))

def create_feature_importance_table (important_features):
    col1 = []
    col2 = []
    for (label, importance) in important_features:
        col1.append(label)
        col2.append(importance)
    datasource = ColumnDataSource({"label": col1, "importance": col2})
    columns = [TableColumn(field="label", title="label"), TableColumn(field="importance", title="importance")]
    return DataTable(source=datasource, columns=columns)


def create_histogram(feature, bins):
    p = figure(title=(feature + " histogram"))
    trimmedFeatureSet = df.sort_values(by=feature).tail(int(size-(size/10)))
    hist, edges = histogram(trimmedFeatureSet[feature], density=False, bins=bins)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.4)
    return p



def create_summary_stats_table(current_feature_name):
    stats = df[current_feature_name].describe().to_dict()
    col1=[]
    col2=[]
    for k, v in stats.items():
        col1.append(k)
        col2.append(v)
    datasource =  ColumnDataSource({"statistic":col1, "value": col2})
    columns = [TableColumn(field="statistic", title="statistic"), TableColumn(field="value", title="value")]
    return DataTable(source=datasource, columns=columns)


if __name__ == "__main__":
    app.run()
