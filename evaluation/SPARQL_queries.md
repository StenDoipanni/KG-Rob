# SPARQL Queries for Knowledge Graph Investigation

1. **Which Situations exist and who are their participants?**
```sparql
PREFIX ns1: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX ns5: <file://./log_enriched.owl#>
PREFIX ns8: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>

SELECT DISTINCT ?situation ?participant
WHERE {
    ?situation a ns1:Situation .
    ?situation ns5:hasParticipant ?participant .
}
```

2. **How many occurrences of DUL:Organism are there?**
```sparql
PREFIX ns1: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX ns8: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>

SELECT (COUNT(DISTINCT ?organism) as ?count)
WHERE {
    ?organism a ns1:Organism .
}
```

3. **What is the average risk reduction value of situations that reduce risk?**
```sparql
PREFIX ns5: <file://./log_enriched.owl#>

SELECT (AVG(?riskValue) as ?averageRiskReduction)
WHERE {
    ?situation ns5:hasRiskReductionValue ?riskValue .
}
```

4. **What are the actions from Khafre involved in the most dangerous avoiding situations?**
```sparql
PREFIX ns5: <file://./log_enriched.owl#>
PREFIX ns8: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>
PREFIX ns9: <file://./log_recommender_system.owl#>

SELECT DISTINCT ?action ?requires
WHERE {
    ?action ns5:isDangerAvoidanceAction true .
    ?action ns9:requires ?requires .
    ?action ns5:preventsEvent ?danger .
    ?danger ns5:dangerLevel "high" .
}
```

5. **What are all the roles that Charlie Chaplin (man_1) has in the scene?**
```sparql
PREFIX ns8: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>
PREFIX ns9: <file://./log_recommender_system.owl#>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT ?role
WHERE {
    ns8:man_1 schema:name "Charlie_Chaplin" .
    ns8:man_1 ns1:hasRole ?role .
}
```

6. **What are all the physical objects and their qualities in the scene?**
```sparql
PREFIX ns1: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX ns8: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>

SELECT DISTINCT ?object ?quality
WHERE {
    ?object a ns1:PhysicalObject .
    ?object ns1:hasQuality ?quality .
}
```

7. **What are all the events that precede other events?**
```sparql
PREFIX ns1: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>

SELECT DISTINCT ?event1 ?event2
WHERE {
    ?event1 ns1:precedes ?event2 .
}
```

8. **What are all the dangerous situations and their associated risk levels?**
```sparql
PREFIX ns5: <file://./log_enriched.owl#>

SELECT DISTINCT ?situation ?dangerLevel
WHERE {
    ?situation ns5:dangerLevel ?dangerLevel .
}
```

9. **What are all the objects that have multiple roles in the scene?**
```sparql
PREFIX ns1: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX ns9: <file://./log_recommender_system.owl#>

SELECT ?object (COUNT(DISTINCT ?role) as ?roleCount)
WHERE {
    ?object ns1:hasRole ?role .
}
GROUP BY ?object
HAVING (COUNT(DISTINCT ?role) > 1)
```

10. **What are all the events that involve both the lion and the man?**
```sparql
PREFIX ns5: <file://./log_enriched.owl#>
PREFIX ns8: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#>

SELECT DISTINCT ?event
WHERE {
    ?event ns5:hasParticipant ns8:man_1 .
    ?event ns5:hasParticipant ns8:lion_3 .
}
```
