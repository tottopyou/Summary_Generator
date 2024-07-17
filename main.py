from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def generate_summary(text, max_length=200, min_length=50, do_sample=False):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, do_sample=do_sample,
                                 early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


input_text = """
More than half of large firms surveyed say they plan to use AI in the next year to do the work previously done by humans. Maria Korneeva/Moment RF/Getty Images
New York
CNN
 — 
Corporate America is rapidly adopting artificial intelligence to automate work once exclusively done by humans.

More than half (61%) of large US firms plan to use AI within the next year to automate tasks previously done by employees, according to a survey of finance chiefs released Thursday.

Those tasks include everything from paying suppliers and doing invoices to financial reporting, said the survey conducted by Duke University and the Federal Reserve Banks of Atlanta and Richmond.

That’s in addition to creative tasks for which some businesses are already relying on ChatGPT and other AI chatbots to assist, including crafting job posts, writing press releases and building marketing campaigns.

The findings show companies are increasingly turning to AI to cut costs, boost profits and make their workers more productive.

“You can’t be running an innovative company without seriously considering these technologies. You run the risk of being left behind,” Duke finance professor John Graham, academic director of the survey, told CNN in a phone interview.

The CFO Survey, a collaboration of Duke and the Atlanta and Richmond Fed banks, found that nearly one in three (32%) firms — large or small — plan to use AI in the next year to complete tasks once done by humans.

Why bosses are deploying AI
Some of this is already happening — especially among larger firms that have the financial firepower to experiment with AI.

Nearly 60% of all companies (and 84% of large companies) surveyed said that over the past year they have already leaned on software, equipment or technology including AI to automate tasks employees previously did. The survey was conducted between May 13 and June 3.

Bosses are turning to AI for a variety of reasons, including to trim what they are spending on human workers.

The CFO Survey found that companies say they are using automation to increase product quality (58% of firms); increase output (49%), reduce labor costs (47%) and substitute for workers (33%).

Still, the good news for workers is that some experts don’t believe AI will cause mass job loss, at least not right away.

“I don’t think there will be a lot of job loss in the year,” said Graham. “In the short run, this will be more about plugging some holes and possibly not hiring someone they would have otherwise — but not laying someone off. In part that’s because this is all-brand new.”

AI co-pilots on the way?
Yet workers will feel the impact of AI adoption, if they aren’t already.

“This could give humans more time to prioritize what is most important and rewarding,” said Graham.

Reid Hoffman, the billionaire investor and co-founder of LinkedIn, told CNN that AI will likely disrupt some jobs but not in the immediate future.

“Years, not decades, but years, not months,” Hoffman said, referring to the timing of AI displacing humans. “I believe in three to five years, we’ll all have kind of an agent co-pilot that’s helping us with anything from how we cook dinner…to doing your job and writing and so forth.”

Hoffman, who last year wrote a book called “Impromptu: Amplifying Our Humanity Through AI” with the assistance from ChatGPT-4, stressed that for a number of years it will be a co-pilot, not a pilot.

“It’s job transformation. Human jobs will be replaced — but will be replaced by other humans using AI,” he said. “The whole ideas is to be the human who is using AI, to learn it, to do it, to make it happen.”

AI and inflation
For now, bosses and employees remain concerned about the cost of living and inflationary pressures.

The CFO Survey found that inflation is the No. 2 concern for the next year among US chief financial officers – behind only the related concern of interest rates and monetary policy.

Most CFOs (57%) expect the price of their products to increase this year at a faster-than-normal pace.

However, there was a divergence in the inflation outlook based on technological adoption. The survey found that companies that implemented automation over the past 12 months expect slower price hikes than those that hadn’t.

Graham, the Duke professor, said that AI could eventually help moderate price increases but isn’t optimistic it will be a major force to easing inflation right now.

“It doesn’t feel like it will be the cure in the next year,” he said.

‘Significant risks’
The CFO survey shows how fast companies are turning to AI — even as safeguards and regulatory frameworks are still being cobbled together.

The rapid adoption of AI in some industries like finance has concerned some.

Treasury Secretary Janet Yellen warned in a speech earlier this month that the use of AI by financial companies poses both “tremendous opportunities and significant risks.”

A report issued last week by Democratic Sen. Gary Peters, chairman of the Homeland Security and Government Affairs Committee, found that exiting regulation “insufficiently addresses” how hedge funds are already using AI.

The report warned that there are “no regulations or requirements” mandating “when and whether a human must be involved in decision making, including related to trading decisions.”

Graham, the Duke professor, said it would be wise for companies in all industries to have strong risk management systems and redundancies in place as they experiment with AI.

“There has been rapid adoption of AI,” he said. “I hope it’s being done with a grain of salt. There will be some situations where companies have embarrassing products or supply chain situations because they moved a little too fast.”
"""

summary = generate_summary(input_text)
print(summary)
