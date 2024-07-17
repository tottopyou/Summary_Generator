# AI Text Summarization with Hugging Face Transformers

This project provides a solution for generating short summaries of large texts, such as scientific articles or news, using the Hugging Face Transformers library. The model used is `BART`, which is well-suited for text summarization tasks.

## Overview

This repository contains a Python script that leverages the `facebook/bart-large-cnn` model from the Hugging Face library to generate concise summaries of large input texts. The model can help automate the process of distilling information from lengthy documents, making it a valuable tool for researchers, journalists, and anyone who deals with large volumes of text.

## Features

- Summarizes large texts into concise summaries.
- Utilizes the pre-trained `BART` model from Hugging Face.
- Easy to use and integrate into other projects.

## Installation

To get started, you need to install the required libraries. Use the following commands to set up your environment:

```bash
pip install transformers
pip install torch
``` 

## Test 

I just decide to use the CNN article about AI: [Link](https://edition.cnn.com/2024/06/20/business/ai-jobs-workers-replacing/index.html)

## Result

The generated summary of the article is as follows:

```
More than half of large US firms plan to use AI within the next year to automate tasks previously done by employees.
Those tasks include everything from paying suppliers and doing invoices to financial reporting.
The findings show companies are increasingly turning to AI to cut costs, boost profits and make their workers more productive.
```

