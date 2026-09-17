//! Attribute macros that validate GemmaTune declarations at compile time.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parser, punctuated::Punctuated, Expr, ExprLit, Lit, Meta, Token};

fn values(input: TokenStream) -> Result<Vec<Meta>, syn::Error> {
    Punctuated::<Meta, Token![,]>::parse_terminated
        .parse(input)
        .map(|items| items.into_iter().collect())
}

fn string_arg(args: &[Meta], name: &str) -> Result<Option<String>, syn::Error> {
    for meta in args {
        if let Meta::NameValue(pair) = meta {
            if pair.path.is_ident(name) {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(value),
                    ..
                }) = &pair.value
                {
                    return Ok(Some(value.value()));
                }
                return Err(syn::Error::new_spanned(
                    &pair.value,
                    format!("`{name}` must be a string"),
                ));
            }
        }
    }
    Ok(None)
}

fn number_arg(args: &[Meta], name: &str) -> Result<Option<f64>, syn::Error> {
    for meta in args {
        if let Meta::NameValue(pair) = meta {
            if pair.path.is_ident(name) {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Float(value),
                    ..
                }) = &pair.value
                {
                    return Ok(Some(value.base10_parse()?));
                }
                if let Expr::Lit(ExprLit {
                    lit: Lit::Int(value),
                    ..
                }) = &pair.value
                {
                    return Ok(Some(value.base10_parse()?));
                }
                return Err(syn::Error::new_spanned(
                    &pair.value,
                    format!("`{name}` must be a number"),
                ));
            }
        }
    }
    Ok(None)
}

fn bool_arg(args: &[Meta], name: &str) -> Result<Option<bool>, syn::Error> {
    for meta in args {
        if let Meta::NameValue(pair) = meta {
            if pair.path.is_ident(name) {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Bool(value),
                    ..
                }) = &pair.value
                {
                    return Ok(Some(value.value));
                }
                return Err(syn::Error::new_spanned(
                    &pair.value,
                    format!("`{name}` must be a boolean"),
                ));
            }
        }
    }
    Ok(None)
}

fn compile_error(error: syn::Error, item: TokenStream) -> TokenStream {
    let error = error.to_compile_error();
    let item = proc_macro2::TokenStream::from(item);
    quote!(#error #item).into()
}

/// Validates a Gemma model declaration.
#[proc_macro_attribute]
pub fn gemma_model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = (|| -> Result<(), syn::Error> {
        let args = values(attr)?;
        let checkpoint = string_arg(&args, "checkpoint")?.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `checkpoint`")
        })?;
        if !matches!(checkpoint.as_str(), "gemma-3-1b-it" | "gemma-3-4b-it") {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "checkpoint must be gemma-3-1b-it or gemma-3-4b-it",
            ));
        }
        if string_arg(&args, "method")?.as_deref() != Some("lora") {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "method must be `lora`",
            ));
        }
        match string_arg(&args, "device")?.as_deref() {
            Some("cpu" | "metal" | "cuda") => Ok(()),
            _ => Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "device must be cpu, metal, or cuda",
            )),
        }
    })();
    match result {
        Ok(()) => item,
        Err(error) => compile_error(error, item),
    }
}

/// Validates a supported training-dataset declaration.
#[proc_macro_attribute]
pub fn training_dataset(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = (|| -> Result<(), syn::Error> {
        let args = values(attr)?;
        if string_arg(&args, "format")?.as_deref() != Some("chat") {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "format must be `chat`",
            ));
        }
        let split = number_arg(&args, "train_split")?.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `train_split`")
        })?;
        if !(0.0..1.0).contains(&split) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "train_split must be between 0 and 1",
            ));
        }
        bool_arg(&args, "redact_pii")?.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `redact_pii`")
        })?;
        Ok(())
    })();
    match result {
        Ok(()) => item,
        Err(error) => compile_error(error, item),
    }
}

/// Validates an LoRA fine-tuning declaration.
#[proc_macro_attribute]
pub fn fine_tune(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = (|| -> Result<(), syn::Error> {
        let args = values(attr)?;
        for name in ["rank", "alpha", "epochs", "learning_rate"] {
            let value = number_arg(&args, name)?.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), format!("missing `{name}`"))
            })?;
            if !value.is_finite() || value <= 0.0 {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("`{name}` must be positive"),
                ));
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => item,
        Err(error) => compile_error(error, item),
    }
}
