import json
import pandas
import numpy
from matplotlib import pyplot
from dateutil.tz import tzlocal
import shutil

confidence_threshold = 0.4

from datetime import datetime, timedelta

def load_birdnet_log(path, confidence_threshold: float = 0.25) -> pandas.DataFrame:
    lines = [json.loads(s) for s in open(path)]
    raw = pandas.DataFrame.from_records(lines)

    raw = raw[raw['msg'] == 'success'].copy()
    raw = raw.explode(['results'])
    raw['timestamp'] = pandas.to_datetime(raw['timestamp'])

    raw[['name','confidence']] = pandas.DataFrame(raw.results.tolist(), index=raw.index)
    raw[['species', 'common']] = raw['name'].str.split("_", expand = True)
    raw.drop(columns=['msg', 'results', 'filename', 'oldest', 'name', 'skipped', 'hour_of_day'], inplace=True)
    return raw.query('confidence > @confidence_threshold', engine='python').copy()

raw = load_birdnet_log('logs/birdnet.log')

def barplot_species_frequency(
    data: pandas.DataFrame,
    legend_count: int = 20,
):
    df = data.query('confidence > @confidence_threshold', engine='python')
    # Plot top species
    ax = df.common.value_counts().nlargest(20).plot(kind='barh')
    return (ax, df)


def roseplot_species_by_minute(
    data,
    timestamp = datetime.now(tzlocal()),
):
    mintime = timestamp - timedelta(minutes=59)
    df = data.query('timestamp >= @mintime and timestamp <= @timestamp', engine='python')

    species = df.common.unique()

    # Count by minute
    species_by_minute = (
        df
        .assign(hod=lambda r: r['timestamp'].dt.minute)
        .groupby(by=['common', 'hod'])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(['hod', 'count'], ascending=[True, False])
    )

    # Transform and reorder
    plotdf = (
        species_by_minute
        .groupby(['hod', 'common'])
        .sum()
        .unstack(level=-1)
        .reset_index(drop=True)
        .fillna(0)
        .reindex([m for m in range(0, 60)], fill_value=0.0)
        # .drop(0, axis='columns')
    )
    plotdf.columns = plotdf.columns.get_level_values('common')

    minutes = [m for m in range(0, 60)]
    minutes_str = []
    for i, m in enumerate(numpy.asarray([f"{_:02}" for _ in minutes])):
        minutes_str.append(m if i % 5 == 0 else "")

    N = len(minutes)
    theta = numpy.linspace(0, 2*numpy.pi, N, endpoint=False)
    width = numpy.pi * 2 / N

    fig, ax = pyplot.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    # ax.set_title(title)
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(plotdf.columns[:-1], plotdf.columns[1:])):
        if n == 0:
            # first column only
            ax.bar(theta, plotdf[c1].values, 
                    width=width,
                    # color=palette[0],
                    edgecolor='none',
                    label=c1,
                    linewidth=0)

        # all other columns
        ax.bar(theta, plotdf[c2].values, 
                width=width, 
                bottom=plotdf.cumsum(axis=1)[c1].values,
                # color=palette[n+1],
                edgecolor='none',
                label=c2,
                linewidth=0)

    leg = ax.legend(loc=(0.8, 0.90), ncol=2)
    ax.set_xticks(theta)
    xtl = ax.set_xticklabels(minutes_str)
    ax.annotate(timestamp.strftime("%Y/%m/%d %H:%M:%S"), xy = (0, 0), xycoords='axes fraction')
    return fig

raw = load_birdnet_log('logs/birdnet.log')
fig = roseplot_species_by_minute(raw)
fig.savefig('species_by_minute.png', bbox_inches="tight")
shutil.copy("species_by_minute.png", "/srv/home-assistant/homeassistant/config/www/birdnet/roseplot_by_minute.png")
