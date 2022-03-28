from scipy import stats
from matplotlib.patches import Ellipse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import seaborn as sns


def pairplot(df, **kwargs):
    """
    散布図行列
    参考) https://qiita.com/shngt/items/b21398cf5e71ef876335
    """
    def corrplot(x, y, data, cmap=None, correlation='pearson', **kwargs):
        if correlation == 'spearman':
            r = stats.spearmanr(data[x], data[y])[0]
        else:
            r = stats.pearsonr(data[x], data[y])[0]
        if cmap is None:
            cmap = 'coolwarm'
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)
        color = cmap((r+1)/2)
        ax.axis('off')
        ax.add_artist(Ellipse((0.5, 0.5), width=np.sqrt(1+r), height=np.sqrt(1-r), angle=45, color=color))
        ax.text(0.5, 0.5, '{:.2f}'.format(r), size='x-large', horizontalalignment='center', verticalalignment='center')

    def crosstabplot(x, y, data, ax, **kwargs):
        cross = pd.crosstab(data[x], data[y]).values
        size = cross / cross.max() * 500
        crosstab_kws = kwargs['crosstab_kws'] if 'crosstab_kws' in kwargs else {}
        scatter_kws = dict(color=sns.color_palette()[0], alpha=0.3)
        scatter_kws.update(crosstab_kws['scatter_kws'] if 'scatter_kws' in crosstab_kws else {})
        text_kws = dict(size='x-large')
        text_kws.update(crosstab_kws['text_kws'] if 'text_kws' in crosstab_kws else {})
        for (xx, yy), count in np.ndenumerate(cross):
            ax.scatter(xx, yy, s=size[xx, yy], **scatter_kws)
            ax.text(xx, yy, count, horizontalalignment='center', verticalalignment='center', **text_kws)

    def catplot(x, y, hue, data, orient, ax, **kwargs):
        box_kws = dict(color='w')
        box_kws.update(kwargs['box_kws'] if 'box_kws' in kwargs else {})
        sns.boxplot(x, y, data=data, orient=orient, ax=ax, **box_kws)
        strip_kws = dict(color=None if hue else sns.color_palette()[0])
        strip_kws.update(kwargs['strip_kws'] if 'strip_kws' in kwargs else {})
        sns.stripplot(x, y, hue, data, orient=orient, ax=ax, **strip_kws)
        legend = ax.get_legend()
        if legend:
            plt.setp(legend, visible=False)

    def scatterplot(x, y, hue, data, ax, **kwargs):
        if hue:
            hues = data[hue].unique()
            colors = sns.color_palette(n_colors=hues.size)
            for h, c in zip(hues, colors):
                ax.scatter(x, y, data=data.query('{col}=={value}'.format(col=hue, value=h)), color=c, **kwargs)
        else:
            ax.scatter(x, y, data=data, **kwargs)

    n_variables = df.columns.size
    hue = kwargs['hue'] if 'hue' in kwargs else None
    figsize = kwargs['figsize'] if 'figsize' in kwargs else figaspect(1) * 0.5 * n_variables
    _, axes = plt.subplots(n_variables, n_variables, figsize=figsize)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(n_variables):
        axes[i, i].get_shared_x_axes().join(*axes[i:n_variables, i])
        if i > 1:
            axes[i, 0].get_shared_y_axes().join(*axes[i, :i-1])

    for (row, col), ax in np.ndenumerate(axes):
        x = df.columns[col]
        y = df.columns[row]
        x_data = df[x]
        y_data = df[y]
        x_dtype = x_data.dtype.name
        y_dtype = y_data.dtype.name
        if x_dtype == 'category':
            x_categories = x_data.cat.categories
        if y_dtype == 'category':
            y_categories = y_data.cat.categories

        if row == col: # diagonal
            hue_data = df[hue] if hue else None
            if x_dtype == 'category':
                bar_kws = dict(alpha=0.4)
                bar_kws.update(kwargs['bar_kws'] if 'bar_kws' in kwargs else {})
                if hue:
                    cross = pd.crosstab(x_data, hue_data)
                    cross.index = cross.index.categories
                    cross.columns = cross.columns.categories if hue_data.dtype.name == 'category' else hue_data.unique()
                    cross.reset_index(inplace=True)
                    melt = pd.melt(cross, id_vars='index',var_name='hue')
                    sns.barplot('index', 'value', 'hue', data=melt, ci=None, orient='v', dodge=False, ax=ax, **bar_kws)
                    plt.setp(ax.get_legend(), visible=False)
                else:
                    cross = pd.crosstab(x_data, []).values.ravel()
                    sns.barplot(x_data.cat.categories, cross, ci=None, orient='v', color=sns.color_palette()[0], ax=ax, **bar_kws)
            else:
                dist_kws = kwargs['dist_kws'] if 'dist_kws' in kwargs else {}
                if hue:
                    hist_kws = dict(color=sns.color_palette(n_colors=hue_data.unique().size), alpha=0.4)
                    hist_kws.update(dist_kws['hist_kws'] if 'hist_kws' in dist_kws else {})
                    hue_values = df[hue].cat.categories if hue_data.dtype.name == 'category' else df[hue].unique()
                    ax.hist([df.query('{hue}=={v}'.format(hue=hue, v=v))[x] for v in hue_values], density=True, histtype='barstacked', **hist_kws)
                    sns.distplot(x_data, hist=False, ax=ax, **dist_kws)
                else:
                    sns.distplot(x_data, ax=ax, **dist_kws)
        elif row < col: # upper
            corr_kws = kwargs['corr_kws'] if 'corr_kws' in kwargs else {}
            corrplot(x, y, data=df, **corr_kws)
        else: # lower
            if x_dtype == 'category' and y_dtype == 'category':
                crosstabplot(x, y, data=df, ax=ax)
            else:
                cat_kws = kwargs['cat_kws'] if 'cat_kws' in kwargs else {}
                if x_dtype == 'category':
                    catplot(x, y, hue, df, 'v', ax, **cat_kws)
                elif y_dtype == 'category':
                    catplot(x, y, hue, df, 'h', ax, **cat_kws)
                else:
                    scatter_kws = kwargs['scatter_kws'] if 'scatter_kws' in kwargs else {}
                    scatterplot(x, y, hue, df, ax, **scatter_kws)
        if row < n_variables-1:
            plt.setp(ax, xlabel='')
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.setp(ax, xlabel=x)
            if x_dtype == 'category':
                plt.setp(ax, xticks=np.arange(x_categories.size), xticklabels=x_data.cat.categories)
        if col > 0:
            plt.setp(ax, ylabel='')
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.setp(ax, ylabel=y)
            if row > 0 and y_dtype == 'category':
                plt.setp(ax, yticks=np.arange(y_categories.size), yticklabels=y_data.cat.categories)
