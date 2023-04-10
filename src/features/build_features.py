# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
    default="data/interim/openmindfulness_contents.csv",
)
@click.argument(
    "output_filepath",
    type=click.Path(),
    default="data/processed/openmindfulness_contents.csv",
)
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn interim data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("making final data set from interim data")

    df = pd.read_csv(input_filepath)

    df = add_features(df)

    df = balance_articles_length(df)

    df.to_csv(output_filepath, index=False)
    return df


def balance_articles_length(df: pd.DataFrame):
    # create a column with number of words in contents column
    df["contents_length"] = df.contents.apply(lambda x: len(x.split()))

    # create a column with number of words in title column
    df["title"] = df.title.fillna("")
    df["title_length"] = df.title.apply(lambda x: len(x.split()))

    df["contents_to_embed"] = df.apply(
        lambda row: row["title"] + "\n" + row["contents"], axis=1
    )
    df["contents_to_embed_length"] = df.contents_to_embed.apply(
        lambda x: len(x.split())
    )

    logger.debug(
        df[["contents_length", "title_length", "contents_to_embed_length"]].describe()
    )
    logger.debug(df.contents_to_embed_length.sum())

    df["merged_paragraphs"] = df["sort_paragraph_nb"].astype(str)

    prev_index = 0

    for index in range(1, len(df)):
        try:
            row = df.loc[index]
            prev_row = df.loc[prev_index]
            if (
                (
                    row["contents_to_embed_length"] < 100
                    or prev_row["contents_to_embed_length"] < 100
                )
                and row["sort_chapter"] == prev_row["sort_chapter"]
                and row["sort_step_nb"] == prev_row["sort_step_nb"]
                and row["sort_section_nb"] == prev_row["sort_section_nb"]
            ):
                df.loc[prev_index, "contents_to_embed"] = (
                    prev_row["contents_to_embed"] + "\n\n" + row["contents_to_embed"]
                )
                df.loc[prev_index, "merged_paragraphs"] = (
                    prev_row["merged_paragraphs"] + " " + str(row["sort_paragraph_nb"])
                )
                df = df.drop(index)
            else:
                prev_index = index
        except Exception as e:
            logger.error(e, exc_info=True)
            raise e

    df["contents_to_embed_length"] = df.contents_to_embed.apply(
        lambda x: len(x.split())
    )
    logger.debug(
        df[["contents_length", "title_length", "contents_to_embed_length"]].describe()
    )
    logger.debug(df.contents_to_embed_length.sum())

    return df


def add_features(df):
    df.loc[df.page_chapter.str.contains("introduction"), "sort_chapter"] = 0
    df.loc[df.page_chapter.str.contains("introduction"), "sort_section_nb"] = df[
        df.page_chapter.str.contains("introduction")
    ].page_chapter.apply(lambda x: x.split("-")[0])

    df.loc[df.page_chapter.str.contains("chapitre-1"), "sort_chapter"] = 1

    df.loc[df.page_chapter.str.contains("chapitre-2"), "sort_chapter"] = 2
    df.loc[df.page_chapter.str.contains("chapitre2"), "sort_chapter"] = 2
    df.loc[df.page_chapter.str.contains("chapitre-2"), "sort_section_nb"] = df[
        df.page_chapter.str.contains("chapitre-2")
    ].page_chapter.apply(lambda x: x.split("-")[0])
    df.loc[df.page_chapter.str.contains("chapitre-2"), "sort_section_title"] = df[
        df.page_chapter.str.contains("chapitre-2")
    ].page_chapter.apply(lambda x: x.split("-")[4:])
    df.loc[df.page_chapter.str.contains("chapitre2"), "sort_section_nb"] = df[
        df.page_chapter.str.contains("chapitre2")
    ].page_chapter.apply(lambda x: x.split("-")[0])
    df.loc[df.page_chapter.str.contains("chapitre2"), "sort_section_title"] = df[
        df.page_chapter.str.contains("chapitre2")
    ].page_chapter.apply(lambda x: x.split("-")[3:])

    df.loc[df.page_chapter.str.contains("conclusion"), "sort_chapter"] = 4
    df.loc[df.page_chapter.str.contains("conclusion"), "sort_section_nb"] = df[
        df.page_chapter.str.contains("conclusion")
    ].page_chapter.apply(lambda x: x.split("-")[0])
    df.loc[df.page_chapter.str.contains("conclusion"), "sort_section_title"] = df[
        df.page_chapter.str.contains("conclusion")
    ].page_chapter.apply(lambda x: x.split("-")[3:])

    df.loc[df.page_chapter.str.contains("expose-68"), "sort_chapter"] = 4
    df.loc[df.page_chapter.str.contains("expose-68"), "sort_section_nb"] = 0
    df.loc[
        df.page_chapter.str.contains("expose-68"), "sort_section_title"
    ] = "expose-68"

    df.loc[df.page_chapter.str.contains("etape-"), "sort_chapter"] = 3
    df.loc[df.page_chapter.str.contains("etape-"), "sort_step_nb"] = df[
        df.page_chapter.str.contains("etape-")
    ].page_chapter.apply(lambda x: x.split("-")[2])
    df.loc[df.page_chapter.str.contains("etape-"), "sort_section_nb"] = df[
        df.page_chapter.str.contains("etape-")
    ].page_chapter.apply(lambda x: x.split("-")[0])
    df.loc[df.page_chapter.str.contains("etape-"), "sort_section_title"] = df[
        df.page_chapter.str.contains("etape-")
    ].page_chapter.apply(lambda x: x.split("-")[3:])

    df["sort_paragraph_nb"] = df["counter"]
    del df["counter"]

    del df["page_chapter"]

    for column in [
        "sort_chapter",
        "sort_step_nb",
        "sort_section_nb",
        "sort_paragraph_nb",
    ]:
        df[column] = df[column].fillna(0).astype(int)
    df["id"] = df.apply(
        lambda row: str(int(row["sort_chapter"]))
        + "."
        + str(int(row["sort_step_nb"]))
        + "."
        + str(int(row["sort_section_nb"])).zfill(2)
        + "."
        + str(int(row["sort_paragraph_nb"])).zfill(2),
        axis=1,
    )

    df = df.sort_values(
        ["sort_chapter", "sort_step_nb", "sort_section_nb", "sort_paragraph_nb"]
    )
    df = df[
        [
            "id",
            "sort_chapter",
            "sort_step_nb",
            "sort_section_nb",
            "sort_paragraph_nb",
            "url",
            "page_title",
            "title",
            "contents",
            "sort_section_title",
        ]
    ]

    df[["sort_chapter", "sort_step_nb", "sort_section_nb", "sort_paragraph_nb"]] = df[
        ["sort_chapter", "sort_step_nb", "sort_section_nb", "sort_paragraph_nb"]
    ].fillna(0)

    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
