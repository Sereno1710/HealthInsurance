# HealthInsurance


Athosss

TO BE DONE - SUGGESTIONS

- in 3.3. Gas usage

    gas_usage: numeric
    customer monthly gas bill amount
        NA 	    	unknown or not applicable
            001 	    Included in rent or in condo fee
            002 	    Included in electricity payment
            003 	    No charge or gas not used
            004..999 	$4 to $999 (Rounded and top-coded)

    group the values in (NA, 001, 002, 003, 004...999) and plot it
        esta logica ainda nao esta a ser apanhada quando fazemos o plot da gas usage distribution

- age: numeric
    customer age in years
    0 Unknown
    1..150 declared age

    a mesma cena para o 0, nao estamos a fazer bem o plot

- correlacoes entre a age e gas_usage, age e health_ins, ...